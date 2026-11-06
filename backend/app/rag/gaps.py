"""Privacy-preserving aggregation of refusal topics in a local SQLite table."""

from __future__ import annotations

import hashlib
import hmac
import re
import sqlite3
import json
from contextlib import closing
from pathlib import Path
from typing import Any

from app.config import settings


_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
_LONG_ID = re.compile(r"\b\d{8,}\b")


def normalize_topic(text: str) -> str:
    """Remove obvious PII and normalize whitespace/punctuation for clustering."""
    value = _EMAIL.sub("[email]", text)
    value = _PHONE.sub("[phone]", value)
    value = _LONG_ID.sub("[id]", value)
    value = re.sub(r"[^a-zA-Z0-9\s]", " ", value.casefold())
    return " ".join(value.split())[:180] or "unspecified topic"


def _privacy_hash(user_id: str) -> str:
    return hmac.new(settings.GAPS_HASH_SALT.encode(), user_id.encode(), hashlib.sha256).hexdigest()


def _connect(path: str | Path | None = None) -> sqlite3.Connection:
    db_path = Path(path or settings.GAPS_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS knowledge_gaps (
        topic_key TEXT NOT NULL, topic_text TEXT NOT NULL, refusal_type TEXT NOT NULL,
        count INTEGER NOT NULL DEFAULT 0, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
        mean_faiss REAL, mean_cross_encoder REAL, samples INTEGER NOT NULL DEFAULT 0,
        user_hashes TEXT NOT NULL DEFAULT '[]', PRIMARY KEY(topic_key, refusal_type)
    )""")
    return conn


def record_refusal(query: str, refusal_type: str, gate_signals: dict[str, Any], user_id: str = "anonymous", db_path: str | Path | None = None) -> None:
    """Upsert one refusal aggregate; raw query/user ID are never stored."""
    import datetime
    if settings.DATABASE_URL and db_path is None:
        _record_postgres(query, refusal_type, gate_signals, user_id)
        return
    topic = normalize_topic(query)
    tokens = set(topic.split())
    topic_key = hashlib.sha256(topic.encode()).hexdigest()
    user_hash = _privacy_hash(user_id)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    faiss = gate_signals.get("faiss_top1")
    ce = gate_signals.get("cross_encoder_max")
    with closing(_connect(db_path)) as conn, conn:
        # Deterministic token-Jaccard clustering merges close paraphrases while
        # retaining a representative topic string for the aggregate.
        for existing_key, existing_topic in conn.execute("SELECT topic_key, topic_text FROM knowledge_gaps WHERE refusal_type=?", (refusal_type,)).fetchall():
            existing_tokens = set(existing_topic.split())
            union = tokens | existing_tokens
            similarity = len(tokens & existing_tokens) / len(union) if union else 1.0
            if similarity >= 0.65:
                topic_key, topic = existing_key, existing_topic
                break
        row = conn.execute("SELECT count, mean_faiss, mean_cross_encoder, samples, user_hashes FROM knowledge_gaps WHERE topic_key=? AND refusal_type=?", (topic_key, refusal_type)).fetchone()
        if row:
            count, old_faiss, old_ce, samples, user_hashes = row
            next_n = samples + 1
            mean_faiss = ((old_faiss or 0) * samples + (faiss or 0)) / next_n if faiss is not None else old_faiss
            mean_ce = ((old_ce or 0) * samples + (ce or 0)) / next_n if ce is not None else old_ce
            hashes = set(json.loads(user_hashes)); hashes.add(user_hash)
            conn.execute("UPDATE knowledge_gaps SET count=?, last_seen=?, mean_faiss=?, mean_cross_encoder=?, samples=?, user_hashes=? WHERE topic_key=? AND refusal_type=?", (count + 1, now, mean_faiss, mean_ce, next_n, json.dumps(sorted(hashes)), topic_key, refusal_type))
        else:
            conn.execute("INSERT INTO knowledge_gaps(topic_key,topic_text,refusal_type,count,first_seen,last_seen,mean_faiss,mean_cross_encoder,samples,user_hashes) VALUES(?,?,?,?,?,?,?,?,?,?)", (topic_key, topic, refusal_type, 1, now, now, faiss, ce, 1, json.dumps([user_hash])))


def ranked_gaps(limit: int = 100, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    if settings.DATABASE_URL and db_path is None:
        import psycopg
        from psycopg.rows import dict_row
        dsn = settings.DATABASE_URL.replace("postgres://", "postgresql://", 1)
        with psycopg.connect(dsn, row_factory=dict_row, connect_timeout=2) as conn:
            conn.execute(_POSTGRES_SCHEMA)
            rows = conn.execute(
                "SELECT topic_text, refusal_type, count, first_seen, last_seen, mean_faiss, mean_cross_encoder FROM knowledge_gaps ORDER BY count DESC, last_seen DESC LIMIT %s",
                (max(1, limit),),
            ).fetchall()
        return [dict(row) for row in rows]
    with closing(_connect(db_path)) as conn, conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT topic_text, refusal_type, count, first_seen, last_seen, mean_faiss, mean_cross_encoder FROM knowledge_gaps ORDER BY count DESC, last_seen DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def record_answer_metadata(user_id: str, query: str, answer_mode: str, gate_scores: dict[str, Any] | None,
                           fence_edited: bool = False, verifier: str | None = None,
                           db_path: str | Path | None = None) -> None:
    """Persist response provenance metadata without retaining messages or raw IDs."""
    import datetime
    topic_hash = hashlib.sha256(normalize_topic(query).encode()).hexdigest()
    user_hash = _privacy_hash(user_id)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = json.dumps(gate_scores or {}, sort_keys=True)
    if settings.DATABASE_URL and db_path is None:
        import psycopg
        from psycopg.types.json import Jsonb
        dsn = settings.DATABASE_URL.replace("postgres://", "postgresql://", 1)
        with psycopg.connect(dsn, connect_timeout=2) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS conversation_decisions (
                id BIGSERIAL PRIMARY KEY, topic_hash TEXT NOT NULL, user_hash TEXT NOT NULL,
                answer_mode TEXT NOT NULL, gate_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
                fence_edited BOOLEAN NOT NULL DEFAULT FALSE, verifier TEXT, created_at TIMESTAMPTZ NOT NULL
            )""")
            conn.execute(
                "INSERT INTO conversation_decisions(topic_hash,user_hash,answer_mode,gate_scores,fence_edited,verifier,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s)",
                (topic_hash, user_hash, answer_mode, Jsonb(gate_scores or {}), fence_edited, verifier, timestamp),
            )
        return
    with closing(_connect(db_path)) as conn, conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS conversation_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, topic_hash TEXT NOT NULL, user_hash TEXT NOT NULL,
            answer_mode TEXT NOT NULL, gate_scores TEXT NOT NULL, fence_edited INTEGER NOT NULL,
            verifier TEXT, created_at TEXT NOT NULL
        )""")
        conn.execute(
            "INSERT INTO conversation_decisions(topic_hash,user_hash,answer_mode,gate_scores,fence_edited,verifier,created_at) VALUES(?,?,?,?,?,?,?)",
            (topic_hash, user_hash, answer_mode, payload, int(fence_edited), verifier, timestamp),
        )


_POSTGRES_SCHEMA = """CREATE TABLE IF NOT EXISTS knowledge_gaps (
    topic_key TEXT NOT NULL, topic_text TEXT NOT NULL, refusal_type TEXT NOT NULL,
    count BIGINT NOT NULL DEFAULT 0, first_seen TIMESTAMPTZ NOT NULL, last_seen TIMESTAMPTZ NOT NULL,
    mean_faiss DOUBLE PRECISION, mean_cross_encoder DOUBLE PRECISION,
    samples BIGINT NOT NULL DEFAULT 0, user_hashes JSONB NOT NULL DEFAULT '[]'::jsonb,
    PRIMARY KEY(topic_key, refusal_type)
)"""


def _record_postgres(query: str, refusal_type: str, gate_signals: dict[str, Any], user_id: str) -> None:
    """Write only normalized topic and one-way user hashes to configured Neon/Postgres."""
    import datetime
    import psycopg
    import json
    from psycopg.types.json import Jsonb
    topic = normalize_topic(query)
    topic_key = hashlib.sha256(topic.encode()).hexdigest()
    tokens = set(topic.split())
    user_hash = _privacy_hash(user_id)
    now = datetime.datetime.now(datetime.timezone.utc)
    faiss_score = gate_signals.get("faiss_top1")
    ce_score = gate_signals.get("cross_encoder_max")
    dsn = settings.DATABASE_URL.replace("postgres://", "postgresql://", 1)
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        conn.execute(_POSTGRES_SCHEMA)
        existing = conn.execute("SELECT topic_key, topic_text FROM knowledge_gaps WHERE refusal_type=%s", (refusal_type,)).fetchall()
        for old_key, old_topic in existing:
            old_tokens = set(old_topic.split())
            similarity = len(tokens & old_tokens) / len(tokens | old_tokens) if tokens | old_tokens else 1.0
            if similarity >= 0.65:
                topic_key, topic = old_key, old_topic
                break
        row = conn.execute(
            "SELECT count, mean_faiss, mean_cross_encoder, samples, user_hashes FROM knowledge_gaps WHERE topic_key=%s AND refusal_type=%s",
            (topic_key, refusal_type),
        ).fetchone()
        if row:
            count, old_faiss, old_ce, samples, user_hashes = row
            n = samples + 1
            mean_faiss = ((old_faiss or 0) * samples + (faiss_score or 0)) / n if faiss_score is not None else old_faiss
            mean_ce = ((old_ce or 0) * samples + (ce_score or 0)) / n if ce_score is not None else old_ce
            hashes = set(user_hashes); hashes.add(user_hash)
            conn.execute(
                "UPDATE knowledge_gaps SET count=%s, last_seen=%s, mean_faiss=%s, mean_cross_encoder=%s, samples=%s, user_hashes=%s WHERE topic_key=%s AND refusal_type=%s",
                (count + 1, now, mean_faiss, mean_ce, n, Jsonb(sorted(hashes)), topic_key, refusal_type),
            )
        else:
            conn.execute(
                "INSERT INTO knowledge_gaps(topic_key,topic_text,refusal_type,count,first_seen,last_seen,mean_faiss,mean_cross_encoder,samples,user_hashes) VALUES(%s,%s,%s,1,%s,%s,%s,%s,1,%s)",
                (topic_key, topic, refusal_type, now, now, faiss_score, ce_score, Jsonb([user_hash])),
            )


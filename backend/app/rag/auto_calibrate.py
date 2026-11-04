"""Deterministic query generation and cluster-specific gate calibration."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

THRESHOLD_VERSION = 1
_TEMPLATES = (
    "What does the guidance say about {topic}?",
    "How can I reduce the risk of {topic}?",
    "What are common signs related to {topic}?",
    "What should families know about {topic}?",
    "{topic} ke baare mein guidance kya hai?",
    "{topic} me kya savdhani rakhni chahiye?",
    "{topic} ke lakshan kya ho sakte hain?",
    "Mujhe {topic} ke baare mein jankari chahiye.",
    "Can you explain {topic} in simple terms?",
    "What prevention steps are recommended for {topic}?",
)


def manifest_hash(metadata: list[dict], source_manifest: str = "") -> str:
    """Hash stable source/chunk identity so changes invalidate calibration."""
    entries = sorted(
        (str(row.get("doc_name", "")), str(row.get("topic", "")), str(row.get("section", "")), str(row.get("chunk_id", "")))
        for row in metadata
    )
    payload = json.dumps({"sources": source_manifest, "chunks": entries}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_in_kb_queries(metadata: list[dict], seed: int = 20261104) -> list[dict[str, str]]:
    """Make reproducible title/section questions, including Hinglish variants."""
    rng = random.Random(seed)
    rows = []
    seen: set[tuple[str, str]] = set()
    for item in metadata:
        topic = str(item.get("topic") or Path(str(item.get("doc_name", "health guidance"))).stem).replace("_", " ")
        section = str(item.get("section") or "general guidance").strip()
        title = section if section.casefold() not in {"general overview", "general", "overview"} else topic
        for template in rng.sample(_TEMPLATES, k=min(4, len(_TEMPLATES))):
            q = template.format(topic=title)
            key = (q.casefold(), str(item.get("doc_name", "")))
            if key not in seen:
                seen.add(key)
                rows.append({"query": q, "label": "in_kb", "doc_name": str(item.get("doc_name", "")), "section": section})
    return rows


def choose_threshold(labels_and_scores: Iterable[tuple[int, float]], fallback: float) -> tuple[float, dict[str, float]]:
    """Select a binary threshold maximizing Youden's J; deterministic ties."""
    rows = [(int(y), float(s)) for y, s in labels_and_scores if np.isfinite(float(s))]
    positives = [score for label, score in rows if label == 1]
    negatives = [score for label, score in rows if label == 0]
    if not positives or not negatives:
        return fallback, {"samples": float(len(rows)), "youden_j": 0.0}
    candidates = sorted({score for _, score in rows})
    candidates = [min(positives + negatives) - 1e-6, *candidates, max(positives + negatives) + 1e-6]
    scored = []
    for threshold in candidates:
        tpr = sum(score >= threshold for score in positives) / len(positives)
        tnr = sum(score < threshold for score in negatives) / len(negatives)
        scored.append((tpr + tnr - 1.0, threshold))
    best_j = max(item[0] for item in scored)
    best_threshold = min(threshold for score, threshold in scored if score == best_j)
    return float(best_threshold), {"samples": float(len(rows)), "youden_j": float(best_j)}


def calibrate_thresholds(
    samples: list[dict],
    scorer: Callable[[str, dict], float],
    *,
    fallback: float = -9.5,
    min_cluster_samples: int = 12,
    seed: int = 20261104,
) -> dict:
    """Fit global and per-document thresholds using only deterministic calibration split."""
    rng = random.Random(seed)
    ordered = list(samples)
    rng.shuffle(ordered)
    split = max(1, int(len(ordered) * 0.70))
    calibration = ordered[:split]
    cluster_rows: dict[str, list[tuple[int, float]]] = {}
    global_rows = []
    for row in calibration:
        score = float(row["score"]) if "score" in row else float(scorer(row["query"], row))
        label = 1 if row["label"] == "in_kb" else 0
        pair = (label, score)
        global_rows.append(pair)
        cluster_rows.setdefault(str(row.get("doc_name") or "__global__"), []).append(pair)
    global_threshold, global_stats = choose_threshold(global_rows, fallback)
    cluster_thresholds = {}
    for cluster, pairs in cluster_rows.items():
        if len(pairs) >= min_cluster_samples and len({label for label, _ in pairs}) == 2:
            threshold, stats = choose_threshold(pairs, global_threshold)
            cluster_thresholds[cluster] = {"threshold": threshold, **stats}
        else:
            cluster_thresholds[cluster] = {"threshold": global_threshold, "samples": float(len(pairs)), "fallback": True}
    return {
        "version": THRESHOLD_VERSION,
        "seed": seed,
        "calibration_count": len(calibration),
        "test_count": len(ordered) - len(calibration),
        "global": {"threshold": global_threshold, **global_stats},
        "clusters": cluster_thresholds,
        "calibration_count": len(calibration),
    }


def load_thresholds(path: str | Path, manifest: str) -> dict | None:
    """Load versioned thresholds only when their source manifest still matches."""
    target = Path(path)
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return data if data.get("version") == THRESHOLD_VERSION and data.get("manifest_hash") == manifest else None
    except (OSError, json.JSONDecodeError):
        return None


def save_thresholds(path: str | Path, manifest: str, calibration: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {**calibration, "manifest_hash": manifest}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def recalibrate_from_evaluation_set(retriever, reranker, thresholds_path: str | Path, manifest: str,
                                   *, faiss_floor: float = 0.45, min_cluster_samples: int = 12) -> dict:
    """Re-fit on the fixed calibration partition only when manifest has changed."""
    eval_file = Path(__file__).resolve().parents[2] / "eval" / "queries.jsonl"
    try:
        from app.rag.coverage import load_uncovered_registry
        from eval.generate_dataset import build_dataset
        rows = build_dataset(retriever.metadata, load_uncovered_registry())
        eval_file.parent.mkdir(parents=True, exist_ok=True)
        eval_file.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Could not generate corpus-derived calibration queries: {exc}") from exc
    rows = [row for row in rows if row.get("split") == "calibration"]
    calibration_rows = []
    for row in rows:
        retrieved = retriever.search(row["query"], top_k=12, min_score=0.0)
        candidates = [item for item in retrieved if float(item.get("score", 0.0)) >= faiss_floor]
        if candidates:
            scores = reranker.predict([(row["query"], str(item.get("chunk_text", ""))[:512]) for item in candidates])
            cluster_idx = int(max(range(len(scores)), key=lambda idx: float(scores[idx])))
            best = candidates[cluster_idx]
            sample_score = float(scores[cluster_idx])
            cluster_name = str(best.get("doc_name", "__global__"))
        else:
            sample_score, cluster_name = -10.0, "__global__"
        calibration_rows.append({**row, "score": sample_score, "doc_name": cluster_name})
    fitted = calibrate_thresholds(calibration_rows, lambda _query, _row: 0.0,
                                  fallback=-9.5, min_cluster_samples=min_cluster_samples)
    save_thresholds(thresholds_path, manifest, fitted)
    return fitted


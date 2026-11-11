"""Index/source manifest consistency and authorization filters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.rag.source_registry import SourceRegistryError, validate_source_registry


def check_integrity(raw_docs_dir: str | Path, vector_store_dir: str | Path) -> dict[str, Any]:
    """Check source hashes, metadata membership and FAISS/metadata cardinality."""
    raw = Path(raw_docs_dir)
    store = Path(vector_store_dir)
    issues: list[str] = []
    try:
        from app.rag.source_registry import read_source_registry
        registry = read_source_registry(raw)
        authorized = set()
        for name, entry in registry.items():
            path = raw / name
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest.startswith(entry["sha256"]):
                authorized.add(name)
            else:
                issues.append(f"source hash mismatch: {name}")
        disk = {path.name for path in raw.iterdir() if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"} and path.name != "SOURCES.md"}
        if disk - set(registry):
            issues.append(f"unregistered source documents: {', '.join(sorted(disk - set(registry)))}")
        if set(registry) - disk:
            issues.append(f"registered source documents missing on disk: {', '.join(sorted(set(registry) - disk))}")
    except (SourceRegistryError, OSError) as exc:
        issues.append(str(exc))
        authorized = set()
    metadata_path = store / "chunk_metadata.json"
    if not metadata_path.is_file():
        metadata_path = store / "metadata.json"
    metadata = []
    if not metadata_path.is_file():
        issues.append("chunk metadata file is missing")
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"chunk metadata cannot be read: {exc}")
    index_path = store / "faiss_index.bin"
    if not index_path.is_file():
        index_path = store / "index.faiss"
    vector_count = None
    if not index_path.is_file():
        issues.append("FAISS index file is missing")
    else:
        try:
            import faiss
            vector_count = int(faiss.read_index(str(index_path)).ntotal)
        except Exception as exc:
            issues.append(f"FAISS index cannot be read: {exc}")
    if vector_count is not None and vector_count != len(metadata):
        issues.append(f"FAISS vector count {vector_count} != metadata rows {len(metadata)}")
    names = {str(row.get("doc_name", "")) for row in metadata}
    stale = sorted(names - authorized)
    absent = sorted(authorized - names)
    if stale:
        issues.append(f"index includes unauthorized source documents: {', '.join(stale)}")
    if absent:
        issues.append(f"registered source documents not indexed: {', '.join(absent)}")
    return {
        "ok": not issues,
        "issues": issues,
        "authorized_documents": sorted(authorized),
        "stale_documents": stale,
        "unindexed_documents": absent,
        "metadata_rows": len(metadata),
        "vector_count": vector_count,
        "manifest_hash": _manifest_hash(raw, metadata),
    }


def _manifest_hash(raw_docs_dir: Path, metadata: list[dict]) -> str:
    """Hash the registered source manifest and ordered chunk identities."""
    from app.rag.auto_calibrate import manifest_hash
    registry_path = raw_docs_dir / "SOURCES.md"
    source_text = registry_path.read_text(encoding="utf-8") if registry_path.is_file() else ""
    return manifest_hash(metadata, source_text)


def filter_authorized_chunks(chunks: list[dict], authorized_documents: set[str]) -> list[dict]:
    """Drop retrieved rows whose document is absent from the trusted manifest."""
    return [row for row in chunks if str(row.get("doc_name", "")) in authorized_documents]


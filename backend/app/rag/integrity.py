"""Index/source manifest consistency and authorization filters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.rag.source_registry import SourceRegistryError, validate_source_registry


def check_integrity(raw_docs_dir: str | Path, vector_store_dir: str | Path) -> dict[str, Any]:
    """Check source hashes, chunk identities and FAISS/metadata compatibility."""
    raw = Path(raw_docs_dir)
    store = Path(vector_store_dir)
    issues: list[str] = []
    index_metadata_compatible = True
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
        index_metadata_compatible = False
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, list):
                issues.append("chunk metadata must be a JSON array")
                metadata = []
                index_metadata_compatible = False
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"chunk metadata cannot be read: {exc}")
            index_metadata_compatible = False
    invalid_rows = [index for index, row in enumerate(metadata) if not isinstance(row, dict)]
    if invalid_rows:
        issues.append(f"chunk metadata contains non-object rows at indexes: {invalid_rows[:10]}")
        index_metadata_compatible = False
    valid_metadata = [row for row in metadata if isinstance(row, dict)]
    chunk_ids = [row.get("chunk_id") for row in valid_metadata]
    if chunk_ids != list(range(len(metadata))):
        issues.append("chunk metadata IDs are not unique sequential indexes")
        index_metadata_compatible = False
    empty_chunks = [row.get("chunk_id", index) for index, row in enumerate(valid_metadata)
                    if not isinstance(row.get("chunk_text"), str) or not row["chunk_text"].strip()]
    if empty_chunks:
        issues.append(f"chunk metadata contains empty text at IDs: {empty_chunks[:10]}")
        index_metadata_compatible = False
    index_path = store / "faiss_index.bin"
    if not index_path.is_file():
        index_path = store / "index.faiss"
    vector_count = None
    if not index_path.is_file():
        issues.append("FAISS index file is missing")
        index_metadata_compatible = False
    else:
        try:
            import faiss
            faiss_index = faiss.read_index(str(index_path))
            vector_count = int(faiss_index.ntotal)
            if faiss_index.d <= 0:
                issues.append("FAISS index has an invalid vector dimension")
                index_metadata_compatible = False
            if faiss_index.metric_type != faiss.METRIC_INNER_PRODUCT:
                issues.append("FAISS index metric is not inner product for normalized cosine search")
                index_metadata_compatible = False
            if vector_count > 0 and faiss_index.d > 0:
                import numpy as np
                batch_size = 4096
                vectors = np.empty((min(batch_size, vector_count), faiss_index.d), dtype=np.float32)
                for start in range(0, vector_count, batch_size):
                    count = min(batch_size, vector_count - start)
                    batch = vectors[:count]
                    faiss_index.reconstruct_n(start, count, batch)
                    norms = np.linalg.norm(batch, axis=1)
                    if not np.isfinite(norms).all() or not np.allclose(norms, 1.0, atol=1e-3):
                        issues.append("FAISS vectors are not finite, non-zero unit vectors")
                        index_metadata_compatible = False
                        break
        except Exception as exc:
            issues.append(f"FAISS index cannot be read: {exc}")
            index_metadata_compatible = False
    if vector_count is not None and vector_count != len(metadata):
        issues.append(f"FAISS vector count {vector_count} != metadata rows {len(metadata)}")
        index_metadata_compatible = False
    names = {str(row.get("doc_name", "")) for row in valid_metadata}
    stale = sorted(names - authorized)
    absent = sorted(authorized - names)
    if stale:
        issues.append(f"index includes unauthorized source documents: {', '.join(stale)}")
    if absent:
        issues.append(f"registered source documents not indexed: {', '.join(absent)}")
    return {
        "ok": not issues,
        "index_metadata_compatible": index_metadata_compatible,
        "issues": issues,
        "authorized_documents": sorted(authorized),
        "stale_documents": stale,
        "unindexed_documents": absent,
        "metadata_rows": len(metadata),
        "vector_count": vector_count,
        "manifest_hash": _manifest_hash(raw, valid_metadata),
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


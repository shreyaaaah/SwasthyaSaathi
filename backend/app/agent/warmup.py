"""Startup warmup for embedding, FAISS, and cross-encoder inference."""

from __future__ import annotations

import threading
import time
from typing import Any

from app.agent.orchestrator import _get_cross_encoder, search_advisories
from app.rag.retriever import get_retriever

_state_lock = threading.Lock()
_state: dict[str, Any] = {"status": "not_started", "duration_ms": None}


def warmup_status() -> dict[str, Any]:
    with _state_lock:
        return dict(_state)


def warm_retrieval_stack() -> dict[str, Any]:
    started = time.monotonic()
    state: dict[str, Any] = {"status": "warming", "embedding_ready": False, "faiss_ready": False, "cross_encoder_ready": False}
    try:
        retriever = get_retriever()
        state["faiss_ready"] = retriever.index is not None and retriever.index.ntotal > 0
        state["embedding_ready"] = retriever.embedding_ready
        if not state["faiss_ready"] or not state["embedding_ready"]:
            raise RuntimeError("Retriever is not ready for query warmup")

        warm_query = "common symptoms and public health guidance"
        chunks = search_advisories(warm_query, top_k=1)
        cross_encoder = _get_cross_encoder()
        if cross_encoder is None:
            raise RuntimeError("Cross-encoder could not be loaded")
        sample = (chunks[0].get("chunk_text") if chunks else "Public health guidance.") or "Public health guidance."
        cross_encoder.predict([(warm_query, sample[:512])])
        state["cross_encoder_ready"] = True
        state["status"] = "ready"
    except Exception as exc:
        state["status"] = "degraded"
        state["error"] = str(exc)[:240]
        print(f"[WARMUP] Retrieval stack warmup failed: {exc}")
    state["duration_ms"] = round((time.monotonic() - started) * 1000, 2)
    with _state_lock:
        _state.clear()
        _state.update(state)
    print(f"[WARMUP] status={state['status']} duration_ms={state['duration_ms']}")
    return dict(state)

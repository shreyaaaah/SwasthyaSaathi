"""Deterministic pre-generation multi-signal response-control gate."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from app.config import settings
from app.rag.coverage import detect_uncovered_condition
from app.rag.intent import Intent, classify_intent


class GateDecision(StrEnum):
    ANSWER = "ANSWER"
    GENERAL = "GENERAL"
    OFF_TOPIC = "OFF_TOPIC"


def decide(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    thresholds: dict | None = None,
    covered_registry: set[str] | None = None,
    uncovered_registry: list[dict] | None = None,
    faiss_floor: float | None = None,
) -> dict[str, Any]:
    """Combine retrieval floor, reranker, per-document threshold and coverage."""
    floor = settings.GATE_FAISS_FLOOR if faiss_floor is None else faiss_floor
    valid = [chunk for chunk in chunks if float(chunk.get("score", 0.0)) >= floor]
    best = max(valid, key=lambda row: float(row.get("ce_score", float("-inf"))), default=None)
    faiss_score = max((float(row.get("score", 0.0)) for row in chunks), default=0.0)
    ce_score = float(best["ce_score"]) if best and best.get("ce_score") is not None else None
    cluster = str(best.get("doc_name", "__global__")) if best else "__global__"
    cluster_entry = (thresholds or {}).get("clusters", {}).get(cluster, {})
    calibrated_global = float((thresholds or {}).get("global", {}).get("threshold", settings.GATE_GLOBAL_THRESHOLD))
    threshold = max(
        settings.GATE_GLOBAL_THRESHOLD,
        calibrated_global,
        float(cluster_entry.get("threshold", calibrated_global)),
    )
    uncovered = detect_uncovered_condition(query, uncovered_registry)
    normalized_query = " " + " ".join("".join(ch if ch.isalnum() else " " for ch in query.casefold()).split()) + " "
    covered_entity = False
    if covered_registry:
        covered_entity = any(len(label) > 3 and f" {label} " in normalized_query for label in covered_registry)
    health_intent = classify_intent(query) == Intent.HEALTH or covered_entity or any(token in query.casefold() for token in (
        "symptom", "disease", "health", "fever", "pain", "infection", "treatment", "medicine", "medical",
        "bimari", "bukhar", "dard", "lakshan", "ilaj", "ilaaj", "swasthya", "sehat",
    ))
    entity_coverage = "uncovered" if uncovered else ("covered" if covered_entity else "no_named_condition")
    if uncovered or health_intent and (not valid or ce_score is None or ce_score < threshold):
        decision, refusal = GateDecision.GENERAL, None
    elif not valid or ce_score is None or ce_score < threshold:
        decision, refusal = GateDecision.OFF_TOPIC, None
    else:
        decision, refusal = GateDecision.ANSWER, None
    return {
        "decision": decision.value,
        "refusal_type": refusal,
        "signals": {
            "faiss_top1": round(faiss_score, 6),
            "faiss_floor": floor,
            "faiss_passed": bool(valid),
            "cross_encoder_max": ce_score,
            "cluster": cluster,
            "cluster_threshold": threshold,
            "entity_coverage": entity_coverage,
            "uncovered_condition": uncovered,
        },
        "valid_chunks": valid,
    }


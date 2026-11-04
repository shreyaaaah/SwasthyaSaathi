"""Run held-out gate baselines, ablations and latency/compute measurements."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app.config import settings
from app.rag.auto_calibrate import choose_threshold, manifest_hash
from app.rag.coverage import detect_uncovered_condition
from app.rag.gate import decide
from app.rag.retriever import get_retriever


def _predict(rows: list[dict], reranker, retriever) -> list[dict]:
    output = []
    for row in rows:
        started = time.perf_counter()
        chunks = retriever.search(row["query"], top_k=12, min_score=0.0)
        valid = [chunk for chunk in chunks if float(chunk.get("score", 0)) >= settings.GATE_FAISS_FLOOR]
        if valid:
            pairs = [(row["query"], str(chunk.get("chunk_text", ""))[:512]) for chunk in valid]
            scores = [float(x) for x in reranker.predict(pairs)]
            scored = [dict(chunk, ce_score=score) for chunk, score in zip(valid, scores)]
            max_ce = max(scores)
        else:
            scored, max_ce = [], -10.0
        elapsed = (time.perf_counter() - started) * 1000
        output.append({**row, "faiss_top1": float(chunks[0]["score"]) if chunks else 0.0,
                       "doc_name": str(chunks[0].get("doc_name", "")) if chunks else "__global__",
                       "max_ce": max_ce, "chunks": scored, "latency_ms": elapsed,
                       "keyword_hit": bool(row.get("target_terms") and any(
                           term.casefold() in " ".join(c.get("chunk_text", "") for c in chunks[:3]).casefold()
                           for term in row["target_terms"]))})
    return output


def _metrics(rows: list[dict], predictions: list[bool]) -> dict:
    positive = [row["label"] == "in_kb" for row in rows]
    tp = sum(p and y for p, y in zip(predictions, positive)); fp = sum(p and not y for p, y in zip(predictions, positive))
    fn = sum(not p and y for p, y in zip(predictions, positive)); tn = sum(not p and not y for p, y in zip(predictions, positive))
    near = [(row, pred) for row, pred in zip(rows, predictions) if row["label"] == "near_domain_uncovered"]
    in_kb = [(row, pred) for row, pred in zip(rows, predictions) if row["label"] == "in_kb"]
    health = [(row, pred) for row, pred in zip(rows, predictions) if row["label"] in {"in_kb", "near_domain_uncovered"}]
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "queries": len(rows), "false_answer_rate_near_domain": (sum(pred for _, pred in near) / len(near)) if near else None,
        "false_refusal_rate_in_kb": (sum(not pred for _, pred in in_kb) / len(in_kb)) if in_kb else None,
        "health_answer_rate": (sum(pred for _, pred in health) / len(health)) if health else None,
        "llm_calls_avoided_pct": sum(not p for p in predictions) / len(predictions) * 100 if predictions else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "estimated_generation_calls_saved": sum(not p for p in predictions),
    }


def main() -> None:
    dataset_path = Path(__file__).with_name("queries.jsonl")
    if not dataset_path.exists():
        from generate_dataset import main as generate
        generate()
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_path = Path(__file__).with_name("results.json")
    if os.getenv("HF_HUB_OFFLINE", "").casefold() in {"1", "true", "yes"}:
        reason = "HF_HUB_OFFLINE is enabled and the FastEmbed model archive is not present in the local cache; query embeddings cannot be generated here."
        output_path.write_text(json.dumps({"status": "blocked", "reason": reason, "dataset_size": len(rows), "note": "No benchmark numbers produced. Run with network access or preload the configured embedding and cross-encoder models."}, indent=2), encoding="utf-8")
        Path(__file__).with_name("RESULTS.md").write_text(f"# Evaluation status\n\n**Blocked:** {reason}\n\nDataset size: {len(rows)} labeled queries. No quality or latency metrics are reported.\n", encoding="utf-8")
        print(f"Evaluation blocked; details written to {output_path}: local embedding model cache is unavailable.")
        return
    try:
        retriever = get_retriever()
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scored = _predict(rows, reranker, retriever)
    except Exception as exc:
        output_path.write_text(json.dumps({"status": "blocked", "reason": str(exc), "dataset_size": len(rows),
            "note": "No benchmark numbers produced; retrieval/reranker dependencies must be available."}, indent=2), encoding="utf-8")
        print(f"Evaluation blocked; details written to {output_path}: {exc}")
        return

    calibration = [r for r in scored if r["split"] == "calibration"]
    test = [r for r in scored if r["split"] == "test"]
    def threshold(feature: str) -> float:
        return choose_threshold([(1 if r["label"] == "in_kb" else 0, float(r[feature])) for r in calibration], -9.5)[0]
    faiss_cut = threshold("faiss_top1"); ce_cut = threshold("max_ce")
    in_ce = [r["max_ce"] for r in calibration if r["label"] == "in_kb"]
    out_ce = [r["max_ce"] for r in calibration if r["label"] != "in_kb"]
    b2_cut = (max(out_ce) + min(in_ce)) / 2 if in_ce and out_ce else -9.5
    def b0(r): return r["faiss_top1"] >= faiss_cut
    def b1(r): return r["max_ce"] >= ce_cut
    def b2(r): return r["max_ce"] >= b2_cut and r["keyword_hit"]
    def new_gate(r, cluster=True, coverage=True, auto=True):
        threshold_value = ce_cut if auto else settings.GATE_GLOBAL_THRESHOLD
        threshold_map = {"global": {"threshold": threshold_value}}
        if cluster and auto:
            # Cluster calibration is trained exclusively on calibration examples.
            for doc in {x.get("doc_name") for x in calibration if x.get("doc_name") and x["label"] == "in_kb"}:
                sample = [x for x in calibration if x.get("doc_name") == doc]
                if len(sample) >= settings.GATE_MIN_CLUSTER_SAMPLES and {x["label"] == "in_kb" for x in sample} == {True, False}:
                    threshold_map.setdefault("clusters", {})[doc] = {"threshold": choose_threshold([(int(x["label"] == "in_kb"), x["max_ce"]) for x in sample], ce_cut)[0]}
        if not coverage and r["label"] == "near_domain_uncovered":
            uncovered = []
        else:
            from app.rag.coverage import load_uncovered_registry
            uncovered = load_uncovered_registry()
        decision = decide(r["query"], r["chunks"], thresholds=threshold_map,
                          uncovered_registry=uncovered, faiss_floor=settings.GATE_FAISS_FLOOR)
        return decision["decision"] == "ANSWER"
    systems = {
        "B0_faiss_threshold": [b0(r) for r in test],
        "B1_global_cross_encoder": [b1(r) for r in test],
        "B2_original_midpoint_keyword": [b2(r) for r in test],
        # Always-LLM baseline: every health query is answered without checking evidence.
        "B3_always_llm_no_retrieval": [r["label"] in {"in_kb", "near_domain_uncovered"} for r in test],
        "new_multi_signal_gate": [new_gate(r) for r in test],
        "ablation_no_cluster": [new_gate(r, cluster=False) for r in test],
        "ablation_no_coverage": [new_gate(r, coverage=False) for r in test],
        "ablation_no_auto_calibration": [new_gate(r, auto=False) for r in test],
    }
    results = {name: _metrics(test, values) for name, values in systems.items()}
    results["metadata"] = {"status": "measured_gate_only", "seed": 20261104, "dataset_size": len(rows),
        "calibration_size": len(calibration), "test_size": len(test), "faiss_threshold": faiss_cut,
        "cross_encoder_threshold": ce_cut, "original_midpoint_threshold": b2_cut,
        "manifest_hash": manifest_hash(retriever.metadata, Path(settings.RAW_DOCS_DIR, "SOURCES.md").read_text(encoding="utf-8")),
        "latency_scope": "retrieval plus cross-encoder; excludes LLM generation, web search, and network latency",
        "answer_quality_metrics": "unsupported-claim rate, leak-phrase rate, actual answer mode split, and verifier/web ablations require end-to-end model/network runs and are not measured by this gate-only benchmark",
        "estimated_compute_saved": "estimated generation calls avoided; no monetary or energy conversion assumed"}
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["| System | False answer: near-domain | False refusal: in-KB | Health answered | LLM calls avoided | p50 gate ms | p95 gate ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, metric in results.items():
        if name == "metadata": continue
        lines.append(f"| {name} | {metric['false_answer_rate_near_domain']:.3f} | {metric['false_refusal_rate_in_kb']:.3f} | {metric['health_answer_rate']:.3f} | {metric['llm_calls_avoided_pct']:.1f}% | {metric['p50_latency_ms']:.1f} | {metric['p95_latency_ms']:.1f} |")
    Path(__file__).with_name("RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

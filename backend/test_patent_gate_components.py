"""Deterministic unit tests for the calibrated evidence gate components."""

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent.orchestrator import run_agent
from app.rag.auto_calibrate import choose_threshold, load_thresholds, manifest_hash, save_thresholds
from app.rag.coverage import detect_uncovered_condition
from app.rag.gaps import normalize_topic, ranked_gaps, record_refusal
from app.rag.gate import decide
from app.rag.integrity import filter_authorized_chunks
from app.rag.intent import Intent, classify_intent, is_immediate_emergency
from eval.generate_dataset import build_dataset


class GateComponentTests(unittest.TestCase):
    def test_multilingual_social_and_health_examples_route_without_llm(self):
        examples = {
            "hi": Intent.GREETING, "hello": Intent.GREETING, "hey there": Intent.GREETING,
            "namaste": Intent.GREETING, "sat sri akal": Intent.GREETING,
            "thanks": Intent.THANKS, "thank u so much": Intent.THANKS,
            "tnx bro": Intent.THANKS, "ok bye": Intent.GOODBYE,
            "mujhe bukhar hai": Intent.HEALTH, "give me the symptoms of chickenpox": Intent.HEALTH,
            "im having temperature": Intent.HEALTH,
            "temperature started 3 days ago, shivering and fluid filled pimples all over my body": Intent.HEALTH,
            "symptoms of dengue": Intent.HEALTH, "what is the capital of France": Intent.OFF_TOPIC,
        }
        for query, expected in examples.items():
            with self.subTest(query=query):
                self.assertEqual(classify_intent(query), expected)
        self.assertEqual(classify_intent("what about children?", [{"role": "user", "content": "dengue"}]), Intent.HEALTH)

    def test_emergency_pattern_detects_chest_pain_and_sweating(self):
        self.assertTrue(is_immediate_emergency("chest pain and sweating"))

    def test_paracetamol_child_dose_is_health_and_not_answered_by_gate_as_grounded(self):
        result = classify_intent("what dose of paracetamol for a 1 year old")
        self.assertEqual(result, Intent.HEALTH)
        gate = decide("what dose of paracetamol for a 1 year old", [], uncovered_registry=[])
        self.assertEqual(gate["decision"], "GENERAL")

    def test_known_uncovered_condition_detected(self):
        found = detect_uncovered_condition("Can you explain chikungunya symptoms?")
        self.assertEqual(found["id"], "chikungunya")

    def test_source_coverage_allows_known_kb_topic(self):
        result = decide("dengue prevention guidance", [{"score": .8, "ce_score": 1.2, "doc_name": "dengue.txt"}],
                        thresholds={"global": {"threshold": 0.0}}, uncovered_registry=[])
        self.assertEqual(result["decision"], "ANSWER")
        self.assertEqual(result["signals"]["cluster"], "dengue.txt")

    def test_uncovered_topic_routes_to_general_even_with_high_relevance(self):
        result = decide("what is treatment for rabies", [{"score": .9, "ce_score": 10, "doc_name": "general.txt"}],
                        thresholds={"global": {"threshold": 0.0}})
        self.assertEqual(result["decision"], "GENERAL")

    def test_cluster_threshold_controls_answer(self):
        row = [{"score": .8, "ce_score": 0.2, "doc_name": "dengue.txt"}]
        thresholds = {"global": {"threshold": -1}, "clusters": {"dengue.txt": {"threshold": .5}}}
        self.assertEqual(decide("dengue prevention", row, thresholds=thresholds, covered_registry={"dengue"}, uncovered_registry=[])["decision"], "GENERAL")

    def test_out_of_domain_query_is_separate_from_health_fallback(self):
        result = decide("write a Python sort function", [{"score": .8, "ce_score": 0, "doc_name": "dengue.txt"}],
                        thresholds={"global": {"threshold": 1.0}}, uncovered_registry=[])
        self.assertEqual(result["decision"], "OFF_TOPIC")

    def test_youden_threshold_separates_sample(self):
        threshold, stats = choose_threshold([(1, 2), (1, 3), (0, -2), (0, -1)], fallback=0)
        self.assertGreater(threshold, -1)
        self.assertEqual(stats["youden_j"], 1.0)

    def test_manifest_stales_saved_threshold_file(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "thresholds.json"
            save_thresholds(path, "abc", {"version": 1, "global": {"threshold": 0.2}})
            self.assertIsNotNone(load_thresholds(path, "abc"))
            self.assertIsNone(load_thresholds(path, "changed"))

    def test_manifest_hash_is_order_independent(self):
        a = [{"doc_name": "a", "chunk_id": 1}, {"doc_name": "b", "chunk_id": 2}]
        self.assertEqual(manifest_hash(a), manifest_hash(list(reversed(a))))

    def test_integrity_filter_rejects_unlisted_source(self):
        rows = [{"doc_name": "allowed.txt"}, {"doc_name": "deleted.txt"}]
        self.assertEqual(filter_authorized_chunks(rows, {"allowed.txt"}), rows[:1])

    def test_gap_logger_strips_pii_and_stores_only_hash(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            db = Path(folder) / "gaps.sqlite3"
            normalized = normalize_topic("rash at person@example.com call +91 98765 43210")
            self.assertNotIn("person@example.com", normalized)
            self.assertNotIn("98765", normalized)
            record_refusal("rash at person@example.com", "uncovered_health_topic", {"faiss_top1": .3}, "user-123", db)
            items = ranked_gaps(db_path=db)
            self.assertEqual(items[0]["count"], 1)
            conn = sqlite3.connect(db)
            try:
                dump = " ".join(str(value) for row in conn.execute("SELECT * FROM knowledge_gaps") for value in row)
            finally:
                conn.close()
            self.assertNotIn("user-123", dump)
            self.assertNotIn("person@example.com", dump)

    def test_gap_logger_clusters_similar_phrasing(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            db = Path(folder) / "gaps.sqlite3"
            record_refusal("chikungunya symptoms please", "uncovered_health_topic", {}, db_path=db)
            record_refusal("chikungunya symptoms", "uncovered_health_topic", {}, db_path=db)
            self.assertEqual(ranked_gaps(db_path=db)[0]["count"], 2)

    def test_dataset_has_minimum_size_and_disjoint_splits(self):
        rows = build_dataset([{"doc_name": "dengue.txt", "topic": "dengue", "section": "prevention"}],
                             [{"id": "rabies", "aliases": ["rabies"], "label": "Rabies"}])
        self.assertGreaterEqual(len(rows), 300)
        self.assertTrue({"calibration", "test"}.issubset({row["split"] for row in rows}))
        self.assertTrue(all(row["label"] != "in_kb" or row["target_doc"] == "dengue.txt" for row in rows))

    def test_gate_routes_uncovered_health_to_fallback_instead_of_refusal(self):
        from app.rag.gate import decide
        result = decide("What is rabies treatment?", [], uncovered_registry=[
            {"id": "rabies", "label": "Rabies", "aliases": ["rabies"]}
        ])
        self.assertEqual(result["decision"], "GENERAL")
        self.assertNotIn("REFUSE", result["decision"])


if __name__ == "__main__":
    unittest.main()

"""Deterministic coverage for latency optimizations and unchanged safety gates."""

import unittest
from unittest.mock import patch

from app.agent.orchestrator import deterministic_triage, deterministic_unsupported_sentences
from app.agent.performance import AnswerCache, normalize_query
from app.main import get_latency_history
from app.config import settings


class PerformanceFeatureTests(unittest.TestCase):
    def test_query_normalization_is_case_and_whitespace_insensitive(self):
        self.assertEqual(normalize_query("  Dengue   Signs "), "dengue signs")

    def test_answer_cache_expires_entries(self):
        cache = AnswerCache(maxsize=2, ttl_seconds=24)
        with patch("app.agent.performance.time.monotonic", side_effect=[10.0, 35.0]):
            cache.put("What are symptoms?", {"answer": "cached"})
            self.assertIsNone(cache.get("What are symptoms?"))

    def test_answer_cache_evicts_least_recently_used(self):
        cache = AnswerCache(maxsize=2, ttl_seconds=24)
        cache.put("one", {"answer": "1"})
        cache.put("two", {"answer": "2"})
        cache.get("one")
        cache.put("three", {"answer": "3"})
        self.assertIsNone(cache.get("two"))
        self.assertEqual(cache.get("one")["answer"], "1")

    def test_emergency_rules_cover_red_flags_without_an_llm(self):
        for query in (
            "I cannot breathe",
            "My face is drooping and speech is slurred",
            "I am bleeding heavily",
            "I want to kill myself",
            "My 2 month old has a fever",
        ):
            with self.subTest(query=query):
                self.assertEqual(deterministic_triage(query)["triage_tag"], "EMERGENCY")

    def test_educational_question_is_deterministically_general(self):
        self.assertEqual(deterministic_triage("What are symptoms of malaria?")["triage_tag"], "GENERAL_INFO")

    def test_uncertain_personal_query_is_left_for_async_triage(self):
        self.assertIsNone(deterministic_triage("I feel unwell"))

    def test_deterministic_grounding_flags_unsupported_sentence(self):
        sources = [{"snippet": "Malaria can cause fever, chills, headache, and fatigue."}]
        flagged = deterministic_unsupported_sentences(
            "Malaria can cause fever, chills, headache, and fatigue. Drink lemon water to cure it.",
            sources,
        )
        self.assertEqual(len(flagged), 1)
        self.assertIn("lemon water", flagged[0])

    def test_latency_endpoint_requires_admin_token(self):
        with patch.object(settings, "ADMIN_TOKEN", "secret"):
            with self.assertRaises(Exception) as error:
                get_latency_history("wrong")
            self.assertEqual(getattr(error.exception, "status_code", None), 401)


if __name__ == "__main__":
    unittest.main()

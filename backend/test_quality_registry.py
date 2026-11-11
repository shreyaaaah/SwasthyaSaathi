"""Fast deterministic checks for source-registry integrity."""

import hashlib
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from app.rag.source_registry import SourceRegistryError, validate_source_registry
from app.agent.orchestrator import (
    enforce_source_backed_fences, is_self_harm_crisis, is_url_allowed, run_agent,
    specificity_gate, sufficiency_check, verify_content_fences, verify_grounding,
)
from app.rag.prompts import limit_follow_up_questions
from fastapi import HTTPException
from app.main import agentic_chat_endpoint, health_check
from app.schemas import ChatRequest
from app.config import settings
from app.rag import retriever as retriever_module


class SourceRegistryTests(unittest.TestCase):
    def make_registry(self, folder: Path, text: str) -> None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        (folder / "guide.txt").write_text(text, encoding="utf-8")
        (folder / "SOURCES.md").write_text(
            "| File | Publisher | Source URL | Review Date | SHA-256 (first 16) | Notes |\n"
            "|------|-----------|------------|-------------|---------------------|-------|\n"
            f"| guide.txt | WHO | https://who.int/ | 2026-11-04 | {digest} | test |\n",
            encoding="utf-8",
        )

    def test_registered_file_and_hash_pass(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder = Path(tmp)
            self.make_registry(folder, "verified text")
            self.assertEqual(list(validate_source_registry(folder)), ["guide.txt"])

    def test_changed_file_fails_hash_check(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder = Path(tmp)
            self.make_registry(folder, "verified text")
            (folder / "guide.txt").write_text("changed text", encoding="utf-8")
            with self.assertRaisesRegex(SourceRegistryError, "hash mismatch"):
                validate_source_registry(folder)

    def test_unregistered_file_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder = Path(tmp)
            self.make_registry(folder, "verified text")
            (folder / "extra.pdf").write_bytes(b"unregistered")
            with self.assertRaisesRegex(SourceRegistryError, "unregistered files"):
                validate_source_registry(folder)

    def test_committed_faiss_index_matches_chunk_metadata(self):
        index_path = Path(settings.VECTOR_STORE_DIR) / "faiss_index.bin"
        metadata_path = Path(settings.VECTOR_STORE_DIR) / "chunk_metadata.json"
        self.assertTrue(index_path.is_file())
        self.assertTrue(metadata_path.is_file())
        index = retriever_module.faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(index.ntotal, len(metadata))


class SafetyAndEvidenceTests(unittest.TestCase):
    def test_response_keeps_at_most_one_follow_up_question(self):
        response = (
            "For now, avoid licking or picking at the area. "
            "Could you tell me a bit more about what you are experiencing? "
            "1. When did it start, and has it been getting worse? "
            "2. Is there swelling or blistering? 3. Did you start a new product?"
        )
        limited = limit_follow_up_questions(response)
        self.assertEqual(limited.count("?"), 1)
        self.assertTrue(limited.rstrip().endswith("?"))
        self.assertIn("When did it start?", limited)
        self.assertNotIn("tell me a bit more", limited.casefold())
        self.assertNotIn("Is there swelling", limited)
        self.assertNotIn("Did you start", limited)

    @patch("app.agent.orchestrator.safe_chat_completion", side_effect=AssertionError("empty-source verifier must not erase general guidance"))
    def test_empty_sources_preserve_general_answer_and_hide_pipeline_details(self, completion):
        draft = "Keep the area clean. The reference sources are empty. Seek care if symptoms worsen."
        result = verify_content_fences(None, "test-model", "dry lips", draft, [])
        self.assertIn("Keep the area clean.", result["answer"])
        self.assertIn("Seek care if symptoms worsen.", result["answer"])
        self.assertNotIn("reference sources", result["answer"].casefold())
        self.assertEqual(result["verifier"], "general_no_source_scope")
        completion.assert_not_called()

    @patch("app.agent.orchestrator.Groq", side_effect=AssertionError("social fast path must not call model"))
    def test_greeting_and_thanks_are_natural_without_sources_or_triage(self, groq):
        for query in ("hi", "thank u so much"):
            with self.subTest(query=query):
                result = run_agent("anonymous", query)
                self.assertEqual(result["answer_mode"], "SOCIAL")
                self.assertFalse(result["sources"])
                self.assertIsNone(result["triage_tag"])
                self.assertNotIn("guideline", result["answer"].casefold())

    @patch("app.agent.orchestrator.record_answer_metadata")
    @patch("app.agent.orchestrator.record_refusal")
    @patch("app.agent.orchestrator.safe_chat_completion")
    @patch("app.agent.orchestrator.Groq", return_value=object())
    @patch("app.agent.orchestrator.triage_classify", return_value={"triage_tag": "GENERAL_INFO"})
    @patch("app.agent.orchestrator.web_search_health_authority", return_value=[])
    @patch("app.agent.orchestrator.search_advisories", return_value=[])
    @patch("app.agent.orchestrator.specificity_gate", return_value={
        "passed": False, "skip_kb": True, "reason": "no local evidence", "faiss_top1": 0,
        "max_ce_score": None, "chunk_ce_scores": [], "valid_chunks": [], "decision": "GENERAL",
        "refusal_type": None, "signals": {"faiss_top1": 0, "cross_encoder_max": None},
    })
    def test_uncovered_health_query_gets_general_answer_without_cold_refusal(
        self, _gate, _search, _web, _triage, _groq, completion, _gap, _metadata,
    ):
        response = SimpleNamespace(model="test-model", choices=[SimpleNamespace(
            message=SimpleNamespace(content="Chickenpox commonly causes an itchy rash with fluid-filled blisters. Seek care for severe symptoms."),
            finish_reason="stop",
        )])
        completion.return_value = response
        result = run_agent("anonymous", "give me the symptoms of chickenpox")
        self.assertEqual(result["answer_mode"], "GENERAL")
        self.assertIn("Chickenpox", result["answer"])
        self.assertTrue(result["answer"].strip())
        self.assertFalse(result["sources"])
        self.assertNotIn("provided reference sources", result["answer"].casefold())
        self.assertIsNone(result.get("refusal_type"))

    @patch("app.agent.orchestrator.record_answer_metadata")
    @patch("app.agent.orchestrator.Groq", side_effect=AssertionError("emergency path must not call model"))
    def test_chest_pain_emergency_leads_with_urgent_care(self, _groq, _record_metadata):
        result = run_agent("anonymous", "chest pain and sweating")
        self.assertEqual(result["triage_tag"], "EMERGENCY")
        self.assertIn("call", result["answer"].casefold())
        self.assertIn(settings.EMERGENCY_NUMBER, result["answer"])
        self.assertTrue(result["sources"])

    def test_source_backed_fence_removes_unsupported_dose(self):
        answer, edited = enforce_source_backed_fences(
            "Rest and drink fluids. Take 500 mg every day.", []
        )
        self.assertTrue(edited)
        self.assertNotIn("500 mg", answer)
        self.assertIn("Rest and drink fluids", answer)

    def test_source_backed_fence_removes_unverified_medicine_recommendation(self):
        answer, edited = enforce_source_backed_fences(
            "Keep the area clean. Take an antibiotic for it.", []
        )
        self.assertTrue(edited)
        self.assertIn("Keep the area clean", answer)
        self.assertNotIn("antibiotic", answer.casefold())

    def test_source_backed_fence_keeps_exactly_supported_dose(self):
        answer, edited = enforce_source_backed_fences(
            "Take 500 mg.", [{"snippet": "The verified source says 500 mg."}]
        )
        self.assertFalse(edited)
        self.assertEqual(answer, "Take 500 mg.")

    def test_web_allowlist_checks_host_boundaries_and_scheme(self):
        self.assertTrue(is_url_allowed("https://www.who.int/health"))
        self.assertFalse(is_url_allowed("https://who.int.attacker.example/health"))
        self.assertFalse(is_url_allowed("file://who.int/etc/passwd"))

    def test_current_self_harm_intent_is_flagged(self):
        self.assertTrue(is_self_harm_crisis("I want to kill myself tonight"))
        self.assertTrue(is_self_harm_crisis("I took too many pills"))
        self.assertTrue(is_self_harm_crisis("I overdosed"))

    def test_educational_question_is_not_crisis_flagged(self):
        self.assertFalse(is_self_harm_crisis("What is suicide prevention?"))

    @patch("app.agent.orchestrator.safe_chat_completion")
    def test_sufficiency_parser_treats_json_false_as_false(self, completion):
        completion.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"is_sufficient": false, "missing_facts": ["dosage"], "reason": "Not covered"}'
        ))])
        result = sufficiency_check(None, "model", "question", [{"doc_name": "test", "chunk_text": "text"}])
        self.assertFalse(result["is_sufficient"])
        self.assertEqual(result["missing_facts"], ["dosage"])

    @patch("app.agent.orchestrator._get_cross_encoder")
    def test_relevance_gate_skips_off_topic_passage(self, get_encoder):
        get_encoder.return_value = SimpleNamespace(predict=lambda pairs: [-10.0 for _ in pairs])
        # A stale/permissive calibration must not weaken the absolute reject floor.
        with patch("app.rag.gate.settings.GATE_GLOBAL_THRESHOLD", -10.0), \
                patch("app.agent.orchestrator.load_thresholds", return_value={"global": {"threshold": -10.0}}):
            gate = specificity_gate("query", [{"score": 0.8, "chunk_text": "unrelated passage"}])
        self.assertTrue(gate["skip_kb"])

    @patch("app.agent.orchestrator.safe_chat_completion")
    def test_grounding_audit_parses_supported_answer(self, completion):
        completion.return_value = SimpleNamespace(
            model="openai/gpt-oss-120b",
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"unsupported_sentences": [], "is_fully_grounded": true, "audit_summary": "Supported"}'), finish_reason="stop")],
        )
        result = verify_grounding(None, "qwen/qwen3.8-27b", "Supported answer.", [{"doc_name": "WHO", "snippet": "Supported fact."}], generator_served_model="qwen/qwen3.8-27b")
        self.assertTrue(result["is_fully_grounded"])

    @patch("app.agent.orchestrator.record_answer_metadata")
    @patch("app.agent.orchestrator.Groq", side_effect=AssertionError("crisis path must not call model"))
    def test_crisis_answer_returns_without_model_call(self, groq, _record_metadata):
        result = run_agent("anonymous", "I might hurt myself tonight")
        self.assertEqual(result["triage_tag"], "EMERGENCY")
        self.assertTrue(result["trace"]["crisis_fast_path"])
        self.assertEqual(len(result["sources"]), 2)

    @patch("app.agent.orchestrator.record_answer_metadata")
    @patch("app.agent.orchestrator.record_refusal")
    @patch("app.agent.orchestrator.Groq", side_effect=AssertionError("medication refusal path must not call model"))
    def test_dose_request_uses_safety_fence_without_model_call(self, groq, _record_gap, _record_metadata):
        result = run_agent("anonymous", "what dose of paracetamol for a 1 year old")
        self.assertTrue(result["trace"]["medication_safety_fast_path"])
        self.assertIn("doctor or pharmacist", result["answer"])
        self.assertNotIn("mg", result["answer"])
        self.assertFalse(result["sources"])


class ApiContractTests(unittest.TestCase):
    def test_health_reports_missing_embedding_model_as_degraded(self):
        retriever = SimpleNamespace(index=SimpleNamespace(ntotal=3341), embedding_ready=False, missing_source_documents=[])
        with patch("app.main.get_retriever", return_value=retriever):
            status = health_check()
        self.assertEqual(status["status"], "degraded")
        self.assertFalse(status["retrieval_ready"])

    def test_health_reports_unindexed_source_documents(self):
        retriever = SimpleNamespace(index=SimpleNamespace(ntotal=3341), embedding_ready=True, missing_source_documents=["guide.txt"])
        with patch("app.main.get_retriever", return_value=retriever):
            status = health_check()
        self.assertFalse(status["source_corpus_ready"])
        self.assertEqual(status["unindexed_documents"], ["guide.txt"])

    def test_chat_endpoint_returns_grounding_and_citation_fields(self):
        agent_result = {
            "answer": "Use official guidance.",
            "sources": [{"doc_name": "WHO", "section": "General", "snippet": "Guidance snippet", "score": 0.8}],
            "is_grounded": True,
            "triage_tag": "GENERAL_INFO",
            "tools_used": ["search_advisories"],
            "response_time_ms": 25.0,
        }
        with patch("app.main.run_agent", return_value=agent_result):
            response = agentic_chat_endpoint(ChatRequest(query="health question"))
        self.assertEqual(response.sources[0].doc_name, "WHO")
        self.assertTrue(response.is_grounded)

    def test_whitespace_only_query_is_rejected(self):
        with self.assertRaises(HTTPException) as context:
            agentic_chat_endpoint(ChatRequest(query="   "))
        self.assertEqual(context.exception.status_code, 400)

if __name__ == "__main__":
    unittest.main()

"""Regression checks for source extraction and bounded RAG chunking."""

import unittest

from app.rag.ingest import _decode_pdf_unicode_glyphs, chunk_document


class ChunkingTests(unittest.TestCase):
    def test_chunks_obey_size_and_overlap_only_on_whole_words(self):
        words = [f"token{index:03d}" for index in range(240)]
        chunks = chunk_document("guide.txt", " ".join(words), chunk_size=120, chunk_overlap=24)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk["chunk_text"]) <= 120 for chunk in chunks))
        allowed_words = set(words)
        for chunk in chunks:
            self.assertTrue(set(chunk["chunk_text"].split()).issubset(allowed_words))
        self.assertTrue(any(
            set(left["chunk_text"].split()) & set(right["chunk_text"].split())
            for left, right in zip(chunks, chunks[1:])
        ))

    def test_section_heading_flushes_previous_content_and_labels_following_chunks(self):
        chunks = chunk_document(
            "guide.txt",
            "INTRODUCTION\nOpening guidance.\nSYMPTOMS\nFever and headache.",
        )

        self.assertEqual([chunk["section"] for chunk in chunks], ["INTRODUCTION", "SYMPTOMS"])
        self.assertIn("Opening guidance", chunks[0]["chunk_text"])
        self.assertNotIn("Fever", chunks[0]["chunk_text"])
        self.assertIn("Fever and headache", chunks[1]["chunk_text"])

    def test_pdf_uni_glyph_sequences_are_decoded(self):
        encoded = "/uni0050/uni0061/uni0072/uni0074/uni0020/uni0068/uni0065/uni0061/uni006C/uni0074/uni0068"
        self.assertEqual(_decode_pdf_unicode_glyphs(encoded), "Part health")

    def test_short_text_without_pdf_glyphs_is_unchanged(self):
        text = "ordinary PDF text /uni0041"
        self.assertEqual(_decode_pdf_unicode_glyphs(text), text)

    def test_invalid_chunk_parameters_fail_early(self):
        with self.assertRaises(ValueError):
            chunk_document("guide.txt", "text", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_document("guide.txt", "text", chunk_size=10, chunk_overlap=10)


if __name__ == "__main__":
    unittest.main()

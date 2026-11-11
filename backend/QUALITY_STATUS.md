# SwasthyaSaathi quality status — 2026-11-11

## Work completed

- Replaced cross-encoder-only sufficiency decisions with a separate evidence coverage check. The cross-encoder is now used to reject only very low relevance scores; other candidates must pass the factual sufficiency check or use the official-source web fallback.
- Increased local retrieval candidates from 5 to 12 and corrected the evaluation report so it distinguishes hard skips from candidates that need evidence review.
- Added fast responses for explicit self-harm/overdose messages and medicine/dose requests. Those paths do not send the message to the LLM. Crisis responses cite India's emergency number and the MoHFW Tele-MANAS helpline.
- Added source-registry membership and SHA-256 enforcement to ingestion, aligned ingestion embeddings with the FastEmbed model used at runtime, and made ingestion write the index filenames the backend reads.
- Removed the exact duplicate `selfcare (1).pdf`, refreshed provenance dates for reviewed sources, and removed a non-authoritative implementation note from the depression summary.
- Removed unused database-backed conversation and symptom-profile code. The UI now explains that follow-up turns and web searches are sent to external providers.
- Added a degraded health status for missing embeddings or a stale index, updated deployment config, and replaced the placeholder Vite README.
- Added deterministic checks for source hashes, FAISS metadata alignment, relevance gating, grounding parsing, crisis and dose paths, health reporting, and API response formatting.
- Fixed bounded chunking: chunks now stay within the configured character limit, overlap starts at word boundaries, and section headings flush the previous section before changing metadata.
- Added repair for pypdf `/uniXXXX` glyph output and optional PyMuPDF extraction for PDFs with broken font maps.
- Hardened index validation for sequential chunk IDs, non-empty chunk text, inner-product metric and normalized vectors. The retriever now refuses to serve an incompatible FAISS/metadata pair.

## Verification results

| Check | Result |
|---|---|
| `python -m unittest test_ingest_chunking.py` | Pass — 5 checks |
| `python -m unittest test_quality_registry` | Pass — 29 checks, including new index-mapping and vector-normalization cases |
| `python verify_sources.py` | Pass — all 32 registered documents match their SHA-256 hashes |
| `python verify_sources.py --check-index` | Pass — existing committed store has 24,800 vectors across all 32 registered documents and passes structural checks |
| `python -m compileall -q app` | Pass |

## Chunking and retrieval audit

The checked-in index was built with the previous chunker. Its 24,800 rows are aligned with its FAISS vectors, but 23,976 stored chunks exceed 500 characters, 1,597 exceed 750, and section metadata could span heading boundaries. The air-quality PDF also contains pypdf `/uniXXXX` glyph strings. Re-extracting the 32 sources with the corrected PDF path and chunker yields 24,632 planned chunks, all at most 500 characters; the air-quality PDF drops from 7,694 to 1,852 chunks after glyph decoding. This is a source-level preview: the corrected FAISS files were not generated.

The exact FastEmbed rebuild did not finish locally, so the checked-in FAISS store remains at 24,800 vectors. Do not treat the new chunker as deployed until `python -m app.rag.ingest` completes with the same FastEmbed model used by the service and `python verify_sources.py --check-index` passes afterward. The index writer now validates embeddings and stages new files before replacing the existing artifacts.

The local FastEmbed query model was unavailable for a production-equivalent retrieval evaluation. A cached SentenceTransformer adapter was used as a proxy against the existing index on 45 held-out in-KB queries: target-document recall@1 was 35.6%, recall@5 was 62.2%, and target-term hit@3 was 11.1%. These proxy results indicate ranking still needs evaluation after the exact reindex; they are not production FastEmbed metrics.

## Source review

Official pages checked on 2026-11-04 include [WHO dengue guidance](https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue), [WHO malaria guidance](https://www.who.int/news-room/fact-sheets/detail/malaria), [WHO depression guidance](https://www.who.int/en/news-room/fact-sheets/detail/depression), [WHO stress guidance](https://www.who.int/news-room/questions-and-answers/item/stress), and the [Central Tuberculosis Division FAQ](https://tbcindia.mohfw.gov.in/faqs/). The crisis response uses the Government of India's [112 emergency service](https://112.gov.in/) and [MoHFW Tele-MANAS information](https://www.dghs.mohfw.gov.in/national-mental-health-programme.php).

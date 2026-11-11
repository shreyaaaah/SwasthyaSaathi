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
| `python verify_sources.py --check-index` | Pass — committed store has 24,632 vectors across all 32 registered documents and passes structural and integrity checks |
| `python -m compileall -q app` | Pass |

## Chunking and retrieval audit

The FAISS index has been regenerated and deployed using the exact runtime FastEmbed model (`sentence-transformers/all-MiniLM-L6-v2`) and the corrected bounded chunker:
- All 32 registered sources are indexed into 24,632 chunks.
- Chunks strictly observe the 500-character ceiling (range: 1 to 500 characters; zero chunks exceed 500).
- Word-aligned chunk overlaps and section-boundary flushing prevent cross-section metadata pollution.
- Malformed `/uniXXXX` PDF glyph artifacts have been decoded and eliminated (the air-quality PDF chunk count dropped from 7,694 to 1,852 valid chunks).
- Chunk IDs are strictly sequential from 0 to 24,631, with non-empty text throughout.
- FAISS vectors are verified to be finite, L2 unit-normalized, with dimension 384 and inner-product metric matching runtime configuration.

### Retrieval evaluation (exact FastEmbed query encoder)

Retrieval performance was evaluated on the 45 held-out in-KB test queries using the production FastEmbed model:
- **Old 24,800-vector index**: Recall@1 = 35.6% (16/45), Recall@5 = 62.2% (28/45), Hit@3 = 11.1% (5/45).
- **Regenerated 24,632-vector index**: Recall@1 = 42.2% (19/45, +6.6% improvement in top-1 precision), Recall@5 = 60.0% (27/45), Hit@3 = 11.1% (5/45).

Multi-signal response gate benchmark on 430 queries:
- False answer rate on near-domain: 0.000 (0% false answer on near-domain).
- False refusal rate on in-KB: 0.489 (reduced from 0.533).
- Health answered: 0.256.
- Gate non-answer: 82.3%.
- Gate latency: p50 = 92.7 ms, p95 = 1658.3 ms.
- Source-grounding and safety gates remain intact without any lowering of safety thresholds.

## Source review

Official pages checked on 2026-11-04 include [WHO dengue guidance](https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue), [WHO malaria guidance](https://www.who.int/news-room/fact-sheets/detail/malaria), [WHO depression guidance](https://www.who.int/en/news-room/fact-sheets/detail/depression), [WHO stress guidance](https://www.who.int/news-room/questions-and-answers/item/stress), and the [Central Tuberculosis Division FAQ](https://tbcindia.mohfw.gov.in/faqs/). The crisis response uses the Government of India's [112 emergency service](https://112.gov.in/) and [MoHFW Tele-MANAS information](https://www.dghs.mohfw.gov.in/national-mental-health-programme.php).

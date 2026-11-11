# SwasthyaSaathi quality status — 2026-11-04

## Work completed

- Replaced cross-encoder-only sufficiency decisions with a separate evidence coverage check. The cross-encoder is now used to reject only very low relevance scores; other candidates must pass the factual sufficiency check or use the official-source web fallback.
- Increased local retrieval candidates from 5 to 12 and corrected the evaluation report so it distinguishes hard skips from candidates that need evidence review.
- Added fast responses for explicit self-harm/overdose messages and medicine/dose requests. Those paths do not send the message to the LLM. Crisis responses cite India's emergency number and the MoHFW Tele-MANAS helpline.
- Added source-registry membership and SHA-256 enforcement to ingestion, aligned ingestion embeddings with the FastEmbed model used at runtime, and made ingestion write the index filenames the backend reads.
- Removed the exact duplicate `selfcare (1).pdf`, refreshed provenance dates for reviewed sources, and removed a non-authoritative implementation note from the depression summary.
- Removed unused database-backed conversation and symptom-profile code. The UI now explains that follow-up turns and web searches are sent to external providers.
- Added a degraded health status for missing embeddings or a stale index, updated deployment config, and replaced the placeholder Vite README.
- Added deterministic checks for source hashes, FAISS metadata alignment, relevance gating, grounding parsing, crisis and dose paths, health reporting, and API response formatting.

## Verification results

| Check | Result |
|---|---|
| `python -m unittest discover -s . -p "test_quality*.py"` | Pass — 15 checks |
| `python verify_sources.py` | Pass — 32 registered documents match their SHA-256 hashes |
| `python verify_sources.py --check-index` | **Needs reindex** — `malaria_guidelines.txt` and `mental_health_depression_guidelines.txt` are missing from the committed index |
| `python -m compileall -q app` and `python -m py_compile run_eval_suite.py verify_sources.py test_quality_registry.py` | Pass |
| Frontend `npm run lint` | Pass |
| Frontend `npm run build` | Pass |
| `git diff --check` | Pass |

The committed FAISS store has 3,341 vectors across 30 documents. The registered source directory now has 32 documents. The existing vector/metadata files are internally aligned, but the two added summaries have not been indexed yet. An ingestion attempt parsed the documents but could not download the FastEmbed model because this environment cannot resolve `huggingface.co`; the index was not changed. Rebuild from `backend` with `python -m app.rag.ingest` in an environment with the model cached or Hugging Face access, then rerun `python verify_sources.py --check-index`.

I attempted a local retrieval smoke check and the updated end-to-end LLM evaluation. Both require the FastEmbed model, which was not cached; model download failed because this environment cannot resolve `huggingface.co`. The evaluation also reported a Groq model-list connection error. The deterministic tests pass, but retrieval and end-to-end behavior are not benchmark-validated until the model is available and the evaluation completes.

## Source review

Official pages checked on 2026-11-04 include [WHO dengue guidance](https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue), [WHO malaria guidance](https://www.who.int/news-room/fact-sheets/detail/malaria), [WHO depression guidance](https://www.who.int/en/news-room/fact-sheets/detail/depression), [WHO stress guidance](https://www.who.int/news-room/questions-and-answers/item/stress), and the [Central Tuberculosis Division FAQ](https://tbcindia.mohfw.gov.in/faqs/). The crisis response uses the Government of India's [112 emergency service](https://112.gov.in/) and [MoHFW Tele-MANAS information](https://www.dghs.mohfw.gov.in/national-mental-health-programme.php).

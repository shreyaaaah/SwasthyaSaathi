---
title: SwasthyaSaathi API
sdk: docker
app_port: 7860
---

# SwasthyaSaathi

SwasthyaSaathi is a public health information chatbot. It searches a locally indexed set of health references and can use web search over an allowlist of health authority sites when the local passages do not cover a question. Answers are informational and do not replace care from a clinician.

## Current capabilities

- React and Vite chat interface, including English voice input where the browser supports it.
- FastAPI endpoints: `GET /`, `GET /health`, and `POST /chat` (`/api/chat` and `/api/query` are aliases).
- FAISS semantic retrieval, a cross-encoder relevance gate when available, an evidence sufficiency check, and an official-source web fallback.
- Emergency triage tags, answer grounding review, source citations, and limited in-session follow-up context.
- No application database or account history. The current question and up to six recent turns are sent to the AI service. When web search is used, the query may be sent to the search provider and source websites. Browser voice recognition may be processed by the browser provider.

The former database-backed longitudinal profile was not connected to the chat API and has been removed. Conversations are not retained by the API.

## Local setup

Requirements: Python 3.10 or newer and Node.js 18 or newer.

### Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Set GROQ_API_KEY in backend/.env
python -m app.rag.ingest
uvicorn app.main:app --reload --port 8000
```

Ingestion verifies that every source document is registered in `data/raw_docs/SOURCES.md` and that its SHA-256 prefix matches before building the FAISS index. The embedding model is downloaded by FastEmbed on first use. Ingestion writes both canonical runtime files (`faiss_index.bin`, `chunk_metadata.json`) and compatibility aliases (`index.faiss`, `metadata.json`) into `backend/data/vector_store/`.

### Frontend

```powershell
cd backend/frontend
npm install
$env:VITE_API_BASE_URL = "http://localhost:8000"
npm run dev
```

Vite prints the local URL. For a production build, run `npm run build` from `backend/frontend`.

## Configuration

Copy `backend/.env.example` to `backend/.env` and set:

- `GROQ_API_KEY`: required for answer generation, triage, sufficiency checks, and grounding review.
- `ALLOWED_ORIGINS`: comma-separated frontend origins; `*` is the development default.
- `VECTOR_STORE_DIR`, `RAW_DOCS_DIR`, and `EMBEDDING_MODEL_NAME`: optional paths and model overrides.

The API does not write conversations to a database. Chat turns provided for follow-up are request data and remain in memory for that request only.

`GET /health` reports vector count, embedding-model readiness, and whether every registered source document is present in the index. A missing embedding model or stale index is reported as `degraded`, not healthy.

## Source maintenance

`backend/data/raw_docs/SOURCES.md` records each document's publisher, source URL, review/fetch date, and SHA-256 prefix. Update it whenever a file is replaced or added. Check local registry and hashes with:

```powershell
cd backend
python verify_sources.py
python verify_sources.py --check-index
```

`--check-index` checks that every registered document is present in the committed FAISS metadata. If it reports missing documents, rebuild with `python -m app.rag.ingest`. Add `--check-urls` to make a best-effort live HTTP status check for registered URLs. A failed URL check needs human review because publishers may move or block automated requests; it does not silently rewrite local source files.

## Quality checks

Run the deterministic checks before a release:

```powershell
cd backend
python -m unittest discover -s . -p "test_quality*.py"
python verify_sources.py
python verify_sources.py --check-index
python -m compileall -q app
cd frontend
npm run lint
npm run build
```

The separate `backend/run_eval_suite.py` is a live, networked LLM evaluation that makes multiple model calls and performs source requests. It requires configured credentials and is intentionally not part of the fast deterministic check.

## Deployment

The root `Dockerfile` builds the FastAPI service on port 7860 and preloads the FastEmbed ONNX embedding model. Configure `GROQ_API_KEY` and `ALLOWED_ORIGINS` in the deployment environment. The production frontend points to `https://swasthyasaathi-3.onrender.com`; local development can use `VITE_API_BASE_URL` to select an API URL. `backend/render.yaml` provides a Render Python service configuration; set the same environment variables there.

## API example

`POST /chat`

```json
{
  "query": "What are warning signs of severe dengue?",
  "conversation_history": []
}
```

The response includes an answer, sources, grounding state, triage tag, tools used, and response time.

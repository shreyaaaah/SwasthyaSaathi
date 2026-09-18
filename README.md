# SwasthyaSaathi (स्वास्थ्य साथी) — Core RAG Public Health Chatbot

SwasthyaSaathi is an end-to-end grounded RAG (Retrieval-Augmented Generation) public health chatbot designed to answer citizen health queries strictly based on official guidelines from the **Ministry of Health and Family Welfare (MoHFW)**, **ICMR**, **NHP**, and **WHO**.

This codebase mirrors the **KrishiRAG** architecture pattern.

---

## 🏗️ Architecture Overview

```
                        +----------------------------+
                        |   MoHFW / ICMR / WHO Docs  |
                        |   (PDFs / Text Guidelines) |
                        +--------------+-------------+
                                       |
                                       v
                        +----------------------------+
                        |    Ingestion Script        |
                        | (ingest.py ~500 tok chunk) |
                        +--------------+-------------+
                                       |
                                       v
                        +----------------------------+
                        | Local FAISS Vector Store   |
                        | + metadata.json sidecar    |
                        +--------------+-------------+
                                       |
User Query ---> Pre-Retrieval Hook ---> Vector Search (top-k) ---> Groq LLM (Grounded System Prompt) ---> Post-Gen Hook ---> Response + Citations
                     |                                                                                        |
                     v                                                                                        v
              [Myth Check Hook]                                                                        [Triage Tag Hook]
                                                                                                              |
                                                                                                              v
                                                                                                   Neon PostgreSQL (conversations DB)
```

---

## 🔌 Extension Points (Decoupled Hooks)

This pass implements the core grounded Q&A chatbot while leaving three clean extension points ready for upcoming features:

1. **Pre-Retrieval Hook (`app/rag/hooks.py` -> `pre_retrieval_hook`)**:
   - Executes before FAISS vector retrieval.
   - Designed for pre-processing query expansion or running myth-check detection to flag medical misinformation prior to RAG retrieval.

2. **Post-Generation Hook (`app/rag/hooks.py` -> `post_generation_hook`)**:
   - Executes after Groq LLM grounded answer generation.
   - Designed for attaching medical triage urgency tags (`EMERGENCY`, `ROUTINE_CONSULT`, `HOME_CARE`) or applying safety disclaimers.

3. **`conversations` PostgreSQL Database Table**:
   - Created on startup in `app/models.py`.
   - Stores `id`, `user_id`, `query`, `response`, `sources`, `triage_tag`, and `timestamp`.
   - Allows adding multi-turn session memory in future passes without database schema migrations.

---

## ⚡ Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.10+
- Node.js 18+

### 2. Backend Setup
```bash
cd backend

# Create & activate virtual environment (optional)
python -m venv venv
# On Windows: venv\Scripts\activate
# On Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure Environment Variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY and DATABASE_URL
```

### 3. Run Document Ingestion (Build FAISS Vector Store)
```bash
python app/rag/ingest.py
```
*Outputs `index.faiss` and `metadata.json` into `backend/data/vector_store/`.*

### 4. Start FastAPI Backend Server
```bash
uvicorn app.main:app --reload --port 8000
```
Backend API will be available at: `http://localhost:8000` (Docs at `http://localhost:8000/docs`).

### 5. Start React Frontend
```bash
cd ../frontend
npm install
npm run dev
```
Frontend will be available at: `http://localhost:5173`.

---

## 🌐 Cloud Deployment (Render + Vercel + Neon)

### 1. Database (Neon PostgreSQL)
- Create a free PostgreSQL project at [Neon.tech](https://neon.tech).
- Copy the connection string: `postgresql://user:pass@ep-xyz.neon.tech/swasthya_db?sslmode=require`.

### 2. Backend Deployment (Render.com)
- Connect repository to **Render Web Service**.
- Use the included `backend/render.yaml` configuration:
  - **Build Command**: `pip install -r requirements.txt && python app/rag/ingest.py`
  - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Environment Variables:
  - `GROQ_API_KEY`: Your Groq API key.
  - `DATABASE_URL`: Your Neon PostgreSQL connection string.

### 3. Frontend Deployment (Vercel)
- Connect `frontend` folder to **Vercel**.
- Build command: `npm run build`
- Output Directory: `dist`
- Environment Variable: `VITE_API_BASE_URL` = `https://your-render-backend-url.onrender.com`.

---

## 📋 API Specifications

### `POST /chat`
- **Request Body**:
  ```json
  {
    "query": "What are the early warning signs of Dengue fever?",
    "user_id": "user_123"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "Primary symptoms of Dengue include high fever (104°F), severe headache, retro-orbital pain, and joint pain...",
    "sources": [
      {
        "doc_name": "dengue_prevention_guidelines.txt",
        "section": "SECTION 2: Clinical Symptoms and Warning Signs",
        "snippet": "High fever with sudden onset, severe headache...",
        "score": 0.7307
      }
    ],
    "is_grounded": true,
    "triage_tag": "GENERAL_HEALTH_INFO"
  }
  ```

### `GET /health`
- Returns vector index status and chunk counts.

---

## 🛡️ License & Disclaimers
This application provides informational public health guidance derived from official MoHFW/WHO guidelines and does not replace professional clinical diagnosis.

# 🩺 SwasthyaSaathi – AI-Powered Public Health Information Assistant

**Reliable health information, grounded in trusted medical sources.**

SwasthyaSaathi is an AI-powered public health information chatbot designed to make reliable health information more accessible. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from a curated collection of trusted health authority documents and generates context-aware responses grounded in verified evidence.

When the local knowledge base does not contain sufficient information, the system can fall back to web search restricted to an allowlist of trusted health authority websites.

The application also incorporates emergency triage indicators, evidence sufficiency checks, source citations, answer grounding verification, and an optional independent second-review mechanism to improve response reliability and transparency.

> **Disclaimer:** SwasthyaSaathi provides general health information for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## ✨ Key Features

- **RAG-Based Question Answering:** Retrieves relevant medical information from a locally indexed knowledge base using semantic search.
- **FAISS Vector Search:** Enables efficient similarity-based retrieval of relevant health document passages.
- **Cross-Encoder Reranking:** Improves retrieval relevance by evaluating and reranking candidate passages.
- **Evidence Sufficiency Check:** Determines whether the retrieved information is sufficient to answer a user's question.
- **Trusted Web Search Fallback:** Searches an allowlist of official health authority websites when local evidence is insufficient.
- **Emergency Triage Detection:** Identifies potential emergency and urgent maternity-related situations and displays appropriate warning indicators.
- **Grounding Verification:** Reviews generated answers to assess whether they are supported by the retrieved evidence.
- **Source Citations:** Displays supporting references and source excerpts to improve transparency and traceability.
- **Optional Second Review:** Provides two independent AI-based reviews focusing on evidence quality and safety/clarity without modifying the original response.
- **Voice Input:** Supports English speech input through browser-based speech recognition where available.
- **Follow-up Context:** Maintains limited conversational context using the current question and up to six recent turns.
- **Privacy-Focused Design:** Does not maintain application database records, user accounts, or persistent conversation history.

---

## 🏗️ System Architecture

SwasthyaSaathi follows a Retrieval-Augmented Generation architecture with additional evidence validation and safety mechanisms.

### High-Level Workflow

1. **User Query:** The user submits a health-related question through the React interface.
2. **API Processing:** The FastAPI backend receives the query and relevant conversation context.
3. **Semantic Retrieval:** The query is converted into an embedding and compared against the FAISS vector index.
4. **Relevance Reranking:** A cross-encoder reranks retrieved passages to identify the most relevant evidence when available.
5. **Evidence Sufficiency:** The system evaluates whether the retrieved passages provide adequate information.
6. **Web Search Fallback:** If local evidence is insufficient, the system can search trusted health authority websites.
7. **Answer Generation:** The Groq-hosted language model generates a response grounded in the available evidence.
8. **Safety and Grounding Review:** The system performs triage detection and evaluates the generated answer's grounding.
9. **Response Delivery:** The frontend displays the answer, citations, grounding information, and relevant urgency indicators.

### Architecture Overview

```text
                 USER
                   |
                   v
          React + Vite Frontend
                   |
                   v
              FastAPI API
                   |
                   v
            Query Processing
                   |
                   v
          Semantic Embeddings
                   |
                   v
             FAISS Search
                   |
                   v
          Cross-Encoder Reranking
                   |
                   v
         Evidence Sufficiency Check
                   |
             +-----+-----+
             |           |
         Sufficient   Insufficient
             |           |
             |           v
             |     Trusted Web Search
             |           |
             +-----+-----+
                   |
                   v
          Groq LLM Generation
                   |
                   v
        Triage & Grounding Review
                   |
                   v
       Answer + Sources + Status
                   |
                   v
             React Frontend
```

---

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript |
| Backend | Python, FastAPI |
| AI / LLM | Groq API |
| Retrieval | RAG, FAISS |
| Embeddings | FastEmbed |
| Reranking | Cross-Encoder |
| Knowledge Base | Curated health authority documents |
| Web Search | Allowlisted official health sources |
| Deployment | Docker, Render |
| Voice Input | Browser Speech Recognition API |

---

## 🔍 Retrieval-Augmented Generation (RAG)

The application uses RAG to reduce dependence on the language model's internal knowledge and provide evidence-grounded health information.

The retrieval pipeline consists of:

- **Document Ingestion:** Health reference documents are collected from trusted publishers and processed into searchable content.
- **Chunking:** Documents are divided into smaller passages for efficient retrieval.
- **Embedding Generation:** FastEmbed converts document passages and user queries into vector representations.
- **Vector Indexing:** FAISS stores and searches the generated embeddings.
- **Similarity Search:** Relevant passages are retrieved based on semantic similarity.
- **Cross-Encoder Reranking:** Retrieved candidates are reranked based on query-passage relevance.
- **Evidence Validation:** The system checks whether the retrieved information is sufficient before generating a response.
- **Grounded Generation:** The language model generates an answer using the available supporting evidence.

### Knowledge Base Management

The source registry is maintained in:

`backend/data/raw_docs/SOURCES.md`

It records:
- Document publisher
- Original source URL
- Review/fetch date
- SHA-256 hash prefix

The ingestion process verifies registered documents and their hashes before generating the FAISS index.

---

## 🛡️ Safety, Reliability & Privacy

Health-related applications require careful handling of information. SwasthyaSaathi incorporates multiple mechanisms to improve response transparency and safety.

### Safety Mechanisms

- Emergency and urgent maternity-related triage indicators.
- Evidence sufficiency checks before answer generation.
- Relevance-based document retrieval and reranking.
- Answer grounding verification.
- Citations and supporting source excerpts.
- Trusted-source-only web fallback.
- Optional independent evidence and safety/clarity reviews.

### Optional Second Review

Users can select **Get a second review** to request an additional assessment of a generated response.

Two independent reviewers operate alongside each other:

1. **Evidence Reviewer:** Evaluates whether the answer is adequately supported by the displayed evidence.
2. **Safety and Clarity Reviewer:** Reviews the response for safety-related concerns and clarity.

Both reviewers use the configured Groq model but perform separate checks.

The review is optional, does not modify the original chatbot response, and is hidden for emergency and urgent-maternity replies.

### Privacy

- No application database or account-based chat history.
- No persistent conversation storage by the API.
- Current questions and up to six recent turns are processed as request data.
- The optional second review sends the question, answer, and displayed source excerpts to the configured Groq model.
- Web-search queries may be shared with the search provider and source websites.
- Browser voice recognition may be processed by the browser provider.

---

## 🚀 Getting Started

### Prerequisites

Make sure you have installed:

- Python 3.10 or newer
- Node.js 18 or newer
- npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/shreyaaaah/SwasthyaSaathi.git
cd SwasthyaSaathi
```

### 2. Backend Setup

Navigate to the backend directory:

```powershell
cd backend
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

Create the environment configuration file:

```powershell
Copy-Item .env.example .env
```

Add your Groq API key to `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key
```

### 3. Build the Vector Index

Run the document ingestion pipeline:

```powershell
python -m app.rag.ingest
```

This process verifies the registered source documents, generates embeddings, and builds the FAISS vector index.

The embedding model is downloaded by FastEmbed during its first use.

Generated files are stored in:

```text
backend/data/vector_store/
```

### 4. Start the Backend Server

```powershell
uvicorn app.main:app --reload --port 8000
```

The backend will be available at:

`http://localhost:8000`

API documentation:

`http://localhost:8000/docs`

### 5. Frontend Setup

Open a new terminal and navigate to the frontend directory:

```powershell
cd backend/frontend
```

Install dependencies:

```bash
npm install
```

Configure the backend URL:

```powershell
$env:VITE_API_BASE_URL = "http://localhost:8000"
```

Start the development server:

```bash
npm run dev
```

Vite will display the frontend URL in the terminal.

---

## ⚙️ Environment Configuration

The backend uses environment variables for configuration.

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key required for LLM-based generation and review |
| `ALLOWED_ORIGINS` | Comma-separated list of permitted frontend origins |
| `VECTOR_STORE_DIR` | Optional vector-store directory override |
| `RAW_DOCS_DIR` | Optional raw-document directory override |
| `EMBEDDING_MODEL_NAME` | Optional embedding model override |

Refer to `backend/.env.example` for the available configuration.

---

## 🔌 API Endpoints

The application exposes the following FastAPI endpoints.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Backend and vector-store health status |
| POST | `/chat` | Main chatbot query endpoint |
| POST | `/api/chat` | Chat endpoint alias |
| POST | `/api/query` | Query endpoint alias |
| POST | `/chat/review` | Optional independent second-review endpoint |

### Example: Chat Request

**POST `/chat`**

Request body:

```json
{
  "query": "What are the warning signs of severe dengue?",
  "conversation_history": []
}
```

The response includes:

- Generated answer
- Supporting sources
- Grounding status
- Triage classification
- Tools used
- Response time

### Health Check

**GET `/health`**

Reports:
- Number of vectors in the index
- Embedding model readiness
- Whether registered source documents are represented in the index
- Overall service health status

A missing embedding model or stale index is reported as `degraded` rather than healthy.

---

## 🧪 Quality Checks

Run the deterministic checks before deployment.

From the backend directory:

```powershell
python -m unittest discover -s . -p "test_quality*.py"
python verify_sources.py
python verify_sources.py --check-index
python -m compileall -q app
```

Run frontend checks:

```powershell
cd frontend
npm run lint
npm run build
```

### Source Verification

Verify document URLs with:

```powershell
python verify_sources.py --check-urls
```

This performs best-effort HTTP status checks. Failed URL checks require manual review and do not automatically modify the registered source documents.

### Live Evaluation

The separate evaluation suite is available at:

```powershell
python run_eval_suite.py
```

It requires configured credentials and performs networked LLM evaluations and source requests. It is separate from the deterministic quality checks.

---

## ☁️ Deployment

SwasthyaSaathi supports Docker-based deployment.

The root `Dockerfile`:

- Builds the FastAPI application.
- Configures the service to run on port `7860`.
- Preloads the FastEmbed ONNX embedding model.

### Deployment Configuration

Set the following environment variables in your deployment platform:

```env
GROQ_API_KEY=your_groq_api_key
ALLOWED_ORIGINS=your_frontend_origin
```

The repository also includes:

`backend/render.yaml`

This provides a Render Python service configuration.

The production frontend is configured to communicate with:

`https://swasthyasaathi-3.onrender.com`

For local development, configure `VITE_API_BASE_URL` with your local backend URL.

---

## 📁 Project Structure

```text
SwasthyaSaathi/
│
├── Dockerfile
├── README.md
│
└── backend/
    ├── app/
    │   ├── main.py
    │   └── rag/
    │       └── ingest.py
    │
    ├── data/
    │   ├── raw_docs/
    │   │   └── SOURCES.md
    │   │
    │   └── vector_store/
    │
    ├── frontend/
    │   ├── src/
    │   ├── public/
    │   ├── package.json
    │   └── vite.config.js
    │
    ├── tests/
    ├── requirements.txt
    ├── .env.example
    ├── render.yaml
    ├── verify_sources.py
    └── run_eval_suite.py
```

*Note: The structure above highlights the main project components. Individual files and modules may vary.*

---

## 🎯 Project Objective

SwasthyaSaathi aims to bridge the gap between accessible health information and reliable medical knowledge by combining semantic search, retrieval-augmented generation, trusted-source fallback, and safety-oriented response validation.

Rather than relying entirely on an LLM's pretrained knowledge, the system prioritizes relevant supporting evidence and makes its sources visible to users.

---

## 👩‍💻 Author

**Shreya**

B.Tech – Computer Science and Engineering (AIML)

[GitHub](https://github.com/shreyaaaah)

---

## 📜 Disclaimer

SwasthyaSaathi is an educational and informational project. Its responses may contain inaccuracies or incomplete information. It does not provide medical diagnoses, prescribe treatments, or replace consultation with qualified healthcare professionals.

For emergencies, contact local emergency services or seek immediate medical attention.

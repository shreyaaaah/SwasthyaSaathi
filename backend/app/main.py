from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import ChatRequest, ChatResponse, SourceMetadata
from app.rag.retriever import get_retriever
from app.agent.orchestrator import run_agent

# Limit PyTorch to 1 thread on CPU-only deploys to reduce RSS significantly
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

app = FastAPI(
    title="SwasthyaSaathi Agentic API",
    description="Public Health Agentic Chatbot grounded in MoHFW, ICMR, NHP, and WHO guidelines.",
    version="2.0.0"
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    # Pre-warm the retriever so the embedding model is loaded at startup
    # rather than on the first request (avoids sudden memory spike on Render).
    try:
        retriever = get_retriever()
        count = retriever.index.ntotal if retriever.index is not None else 0
        print(f"Retriever pre-warmed: {count} vectors loaded.")
    except Exception as e:
        print(f"Retriever pre-warm failed (non-fatal): {e}")

@app.get("/")
def root():
    return {
        "app": "SwasthyaSaathi Agentic Public Health API",
        "status": "online",
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    retriever = get_retriever()
    is_ready = retriever.index is not None and retriever.index.ntotal > 0
    return {
        "status": "healthy" if is_ready else "unindexed",
        "vector_store_ready": is_ready,
        "vectors_count": retriever.index.ntotal if is_ready else 0
    }

@app.post("/chat", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
@app.post("/api/query", response_model=ChatResponse)
def agentic_chat_endpoint(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    user_id = request.user_id or "anonymous"
    agent_output = run_agent(user_id=user_id, query=request.query)

    # Format sources for API response
    raw_sources = agent_output.get("sources", [])
    formatted_sources = [
        SourceMetadata(
            doc_name=s.get("doc_name", "Guideline"),
            section=s.get("section", "General"),
            snippet=s.get("snippet", s.get("chunk_text", ""))[:300] + "...",
            score=s.get("score"),
            source_type=s.get("source_type", "knowledge_base"),
            source_url=s.get("source_url")
        )
        for s in raw_sources
    ]

    return ChatResponse(
        answer=agent_output.get("answer", ""),
        sources=formatted_sources,
        is_grounded=agent_output.get("is_grounded"),
        triage_tag=agent_output.get("triage_tag", "GENERAL_INFO"),
        tools_used=agent_output.get("tools_used", []),
        response_time_ms=agent_output.get("response_time_ms")
    )

import secrets
import threading
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import ChatRequest, ChatResponse, SourceMetadata
from app.rag.retriever import get_retriever
from app.agent.orchestrator import CHAT_BUDGET_SECONDS, run_agent
from app.agent.performance import latency_history
from app.agent.warmup import warm_retrieval_stack, warmup_status
from app.rag.gaps import ranked_gaps

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
    # Warm all local inference components before the service receives chat traffic.
    try:
        warm_retrieval_stack()
    except Exception as e:
        print(f"Model warmup failed (non-fatal): {e}")
        if settings.INTEGRITY_MODE == "strict":
            raise

    # Populate process-local answers for the non-emergency topic chips in the background.
    # The severe chest-pain chip is intentionally never cached.
    def prewarm_chip_answers():
        chip_queries = [
            "What should I do about high fever lasting 3 days?",
            "I've had a persistent cough for 3 weeks",
            "What are early warning signs and home care for Dengue fever?",
            "What antenatal checkups and nutrition are needed during pregnancy?",
            "Do antibiotics cure viral infections like cold or flu?",
            "Does turmeric cure diabetes?",
        ]
        for chip_query in chip_queries:
            try:
                run_agent("cache-prewarm", chip_query, [], budget_seconds=CHAT_BUDGET_SECONDS, cache_only=True)
            except Exception as exc:
                print(f"[CACHE PREWARM] Skipped a topic after failure: {exc}")

    threading.Thread(target=prewarm_chip_answers, name="chip-cache-prewarm", daemon=True).start()

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
    vector_store_ready = retriever.index is not None and retriever.index.ntotal > 0
    embedding_ready = retriever.embedding_ready
    missing_source_documents = retriever.missing_source_documents
    corpus_ready = not missing_source_documents
    integrity = getattr(retriever, "integrity_report", {"ok": True, "issues": []})
    is_ready = vector_store_ready and embedding_ready and corpus_ready and integrity.get("ok", False)
    return {
        "status": "healthy" if is_ready else ("degraded" if vector_store_ready else "unindexed"),
        "vector_store_ready": vector_store_ready,
        "embedding_model_ready": embedding_ready,
        "source_corpus_ready": corpus_ready,
        "unindexed_documents": missing_source_documents,
        "retrieval_ready": is_ready,
        "vectors_count": retriever.index.ntotal if vector_store_ready else 0,
        "integrity_ok": integrity.get("ok", False),
        "integrity_issues": integrity.get("issues", []),
        "index_status": "ready" if integrity.get("ok", False) and vector_store_ready else "unavailable",
        "chunk_count": integrity.get("metadata_rows", 0),
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "manifest_hash": integrity.get("manifest_hash"),
    }


@app.get("/warmup")
def warmup_check():
    """Run the local embedding/retrieval/reranker smoke path and report readiness."""
    status = warmup_status()
    if status.get("status") != "ready":
        status = warm_retrieval_stack()
    return status

@app.post("/chat", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
@app.post("/api/query", response_model=ChatResponse)
def agentic_chat_endpoint(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    user_id = request.user_id or "anonymous"
    agent_output = run_agent(
        user_id=user_id,
        query=request.query,
        conversation_history=[turn.model_dump() for turn in request.conversation_history],
        budget_seconds=CHAT_BUDGET_SECONDS,
    )
    latency_history.add(agent_output.get("response_time_ms"), agent_output.get("timing_breakdown", []))

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
        response_time_ms=agent_output.get("response_time_ms"),
        gate_decision=agent_output.get("gate_decision"),
        refusal_type=agent_output.get("refusal_type"),
        answer_mode=agent_output.get("answer_mode"),
        gate_scores=agent_output.get("gate_scores"),
        intent=agent_output.get("intent"),
        emergency_contact=agent_output.get("emergency_contact"),
    )


@app.get("/latency")
def get_latency_history(x_admin_token: str | None = Header(default=None)):
    """Return recent timing-only diagnostics; never include queries or answers."""
    if not settings.ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Latency diagnostics are not configured")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, settings.ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid admin token")
    records = latency_history.snapshot()
    durations = sorted(
        float(record["response_time_ms"])
        for record in records
        if record.get("response_time_ms") is not None
    )

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))
        return round(values[index], 2)

    return {
        "count": len(records),
        "p50_ms": percentile(durations, 0.50),
        "p95_ms": percentile(durations, 0.95),
        "records": records[-50:],
    }


@app.get("/admin/knowledge-gaps")
def get_knowledge_gaps(limit: int = 100, x_admin_token: str | None = Header(default=None)):
    """Return aggregate refusal topics only; requires the configured admin token."""
    if not settings.ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Knowledge-gap admin endpoint is not configured")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, settings.ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return {"topics": ranked_gaps(max(1, min(limit, 500)))}

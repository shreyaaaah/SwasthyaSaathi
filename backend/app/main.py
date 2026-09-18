import json
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db, init_db
from app.models import Conversation
from app.schemas import ChatRequest, ChatResponse, SourceMetadata, IngestResponse
from app.rag.retriever import get_retriever
from app.agent.orchestrator import run_agent

app = FastAPI(
    title="SwasthyaSaathi Agentic API",
    description="Public Health Agentic Chatbot grounded in MoHFW, ICMR, NHP, and WHO guidelines.",
    version="2.0.0"
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()
    print("Database tables initialized successfully.")

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
        "vectors_count": retriever.index.ntotal if is_ready else 0,
        "database_url": settings.formatted_db_url.split("@")[-1] if "@" in settings.formatted_db_url else "local"
    }

@app.post("/chat", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
@app.post("/api/query", response_model=ChatResponse)
def agentic_chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
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
            score=s.get("score")
        )
        for s in raw_sources
    ]

    # Also log to conversations table
    try:
        conv_record = Conversation(
            user_id=user_id,
            query=request.query,
            response=agent_output.get("answer", ""),
            sources=json.dumps([s.model_dump() for s in formatted_sources], ensure_ascii=False),
            triage_tag=agent_output.get("triage_tag", "GENERAL_INFO")
        )
        db.add(conv_record)
        db.commit()
    except Exception as e:
        print(f"Failed to log conversation record: {e}")
        db.rollback()

    return ChatResponse(
        answer=agent_output.get("answer", ""),
        sources=formatted_sources,
        is_grounded=len(formatted_sources) > 0,
        triage_tag=agent_output.get("triage_tag", "GENERAL_INFO")
    )

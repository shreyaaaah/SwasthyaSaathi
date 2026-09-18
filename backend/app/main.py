import json
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db, init_db
from app.models import Conversation
from app.schemas import ChatRequest, ChatResponse, SourceMetadata, IngestResponse
from app.rag.hooks import pre_retrieval_hook, post_generation_hook
from app.rag.retriever import get_retriever
from app.rag.generator import generate_grounded_answer
from app.rag.ingest import build_and_save_index

app = FastAPI(
    title="SwasthyaSaathi API",
    description="Public Health RAG Chatbot grounded in MoHFW, ICMR, NHP, and WHO guidelines.",
    version="1.0.0"
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
    print("Database tables verified/created successfully.")

@app.get("/")
def root():
    return {
        "app": "SwasthyaSaathi Public Health Chatbot API",
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
def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    # 1. Pre-retrieval Hook (Extension point for myth check / query rewriting)
    pre_hook_result = pre_retrieval_hook(request.query)
    processed_query = pre_hook_result.get("processed_query", request.query)

    # 2. Retrieval Phase
    retriever = get_retriever()
    retrieved_chunks = retriever.search(query=processed_query, top_k=4, min_score=0.15)

    # Format sources metadata
    sources = [
        SourceMetadata(
            doc_name=chunk.get("doc_name", "Unknown Document"),
            section=chunk.get("section", "General"),
            snippet=chunk.get("chunk_text", "")[:300] + "...",
            score=chunk.get("score")
        )
        for chunk in retrieved_chunks
    ]

    # 3. Generation Phase
    answer, is_grounded = generate_grounded_answer(query=processed_query, retrieved_chunks=retrieved_chunks)

    # 4. Post-generation Hook (Extension point for triage tagging)
    post_hook_result = post_generation_hook(answer, retrieved_chunks)
    final_answer = post_hook_result.get("final_answer", answer)
    triage_tag = post_hook_result.get("triage_tag", "GENERAL_HEALTH_INFO")

    # 5. Log to PostgreSQL `conversations` Table
    try:
        sources_json = json.dumps([s.model_dump() for s in sources], ensure_ascii=False)
        conv_record = Conversation(
            user_id=request.user_id or "anonymous",
            query=request.query,
            response=final_answer,
            sources=sources_json,
            triage_tag=triage_tag
        )
        db.add(conv_record)
        db.commit()
    except Exception as e:
        print(f"Failed to log conversation to database: {e}")
        db.rollback()

    return ChatResponse(
        answer=final_answer,
        sources=sources,
        is_grounded=is_grounded,
        triage_tag=triage_tag
    )

@app.post("/ingest", response_model=IngestResponse)
def trigger_ingestion():
    result = build_and_save_index()
    if result.get("status") == "success":
        get_retriever().reload()
        return IngestResponse(
            status="success",
            documents_indexed=result.get("documents_indexed", 0),
            chunks_created=result.get("chunks_created", 0),
            vector_store_path=result.get("vector_store_path", "")
        )
    else:
        raise HTTPException(status_code=500, detail=result.get("message", "Ingestion failed"))

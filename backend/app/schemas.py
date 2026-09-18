from typing import List, Optional
from pydantic import BaseModel

class SourceMetadata(BaseModel):
    doc_name: str
    section: str
    snippet: str
    score: Optional[float] = None

class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = "anonymous"

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata] = []
    is_grounded: bool = True
    triage_tag: Optional[str] = None  # Extension point for triage tagging

class IngestResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_created: int
    vector_store_path: str

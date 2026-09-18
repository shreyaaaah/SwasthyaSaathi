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
    triage_tag: Optional[str] = "GENERAL_INFO"
    tools_used: List[str] = []
    response_time_ms: Optional[float] = None

class IngestResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_created: int
    vector_store_path: str

class SymptomLogSchema(BaseModel):
    id: int
    user_id: str
    query_text: str
    topic: Optional[str] = None
    triage_tag: Optional[str] = None
    created_at: Optional[str] = None

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class SourceMetadata(BaseModel):
    doc_name: str
    section: str
    snippet: str
    score: Optional[float] = None
    source_type: Optional[str] = "knowledge_base"
    source_url: Optional[str] = None

class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    user_id: Optional[str] = "anonymous"
    conversation_history: List[ConversationTurn] = Field(default_factory=list, max_length=8)

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata] = []
    is_grounded: Optional[bool] = None
    triage_tag: Optional[str] = "GENERAL_INFO"
    tools_used: List[str] = []
    response_time_ms: Optional[float] = None

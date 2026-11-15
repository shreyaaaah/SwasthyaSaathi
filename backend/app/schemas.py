from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

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
    gate_decision: Optional[str] = None
    refusal_type: Optional[str] = None
    answer_mode: Optional[str] = None
    gate_scores: Optional[dict] = None
    intent: Optional[str] = None
    emergency_contact: Optional[str] = None


class NearbyHospitalsRequest(BaseModel):
    place: Optional[str] = Field(default=None, min_length=1, max_length=120)
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def validate_location(self):
        has_place = bool(self.place and self.place.strip())
        has_coordinates = self.latitude is not None and self.longitude is not None
        partial_coordinates = (self.latitude is None) != (self.longitude is None)
        if partial_coordinates or has_place == has_coordinates:
            raise ValueError("Provide either a place or both latitude and longitude.")
        if self.place is not None:
            self.place = self.place.strip()
        return self

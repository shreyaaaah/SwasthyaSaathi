from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from app.db import Base

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(255), default="anonymous", index=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON-encoded string of source documents used
    triage_tag = Column(String(50), nullable=True)  # Reserved extension point for future triage tagging
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id='{self.user_id}', timestamp={self.timestamp})>"

class SymptomLog(Base):
    __tablename__ = "symptom_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(255), default="anonymous", index=True)
    query_text = Column(Text, nullable=False)
    topic = Column(String(100), nullable=True)
    triage_tag = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<SymptomLog(id={self.id}, user_id='{self.user_id}', topic='{self.topic}', triage='{self.triage_tag}')>"

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

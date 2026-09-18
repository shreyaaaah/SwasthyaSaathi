from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

db_url = settings.formatted_db_url
is_sqlite = db_url.startswith("sqlite")

engine_kwargs = {
    "connect_args": {"check_same_thread": False} if is_sqlite else {}
}

if not is_sqlite:
    engine_kwargs.update({
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
        "pool_recycle": 300,
    })

engine = create_engine(db_url, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from app.models import Conversation, SymptomLog
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully.")

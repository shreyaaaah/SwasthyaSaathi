import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_FILE):
    load_dotenv(ENV_FILE)
else:
    load_dotenv()

class Settings(BaseSettings):
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'swasthya.db')}").strip()
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", os.path.join(BASE_DIR, "data", "vector_store")).strip()
    RAW_DOCS_DIR: str = os.getenv("RAW_DOCS_DIR", os.path.join(BASE_DIR, "data", "raw_docs")).strip()
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2").strip()
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "qwen/qwen3.8-27b").strip()
    
    @property
    def formatted_db_url(self) -> str:
        url = self.DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url

settings = Settings()

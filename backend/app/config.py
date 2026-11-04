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
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", os.path.join(BASE_DIR, "data", "vector_store")).strip()
    RAW_DOCS_DIR: str = os.getenv("RAW_DOCS_DIR", os.path.join(BASE_DIR, "data", "raw_docs")).strip()
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2").strip()
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "qwen/qwen3.8-27b").strip()
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*").strip()
    GATE_GLOBAL_THRESHOLD: float = float(os.getenv("GATE_GLOBAL_THRESHOLD", "-9.5"))
    GATE_FAISS_FLOOR: float = float(os.getenv("GATE_FAISS_FLOOR", "0.45"))
    GATE_MIN_CLUSTER_SAMPLES: int = int(os.getenv("GATE_MIN_CLUSTER_SAMPLES", "12"))
    UNCOVERED_CONDITIONS_FILE: str = os.getenv(
        "UNCOVERED_CONDITIONS_FILE", os.path.join(BASE_DIR, "data", "uncovered_conditions.json")
    ).strip()
    GAPS_DB_PATH: str = os.getenv("GAPS_DB_PATH", os.path.join(BASE_DIR, "data", "knowledge_gaps.sqlite3")).strip()
    GAPS_HASH_SALT: str = os.getenv("GAPS_HASH_SALT", "local-development-only-change-me").strip()
    ADMIN_TOKEN: str = os.getenv("ADMIN_TOKEN", "").strip()
    INTEGRITY_MODE: str = os.getenv("INTEGRITY_MODE", "warn").strip().lower()
    HELPLINE_REFERRAL_TEXT: str = os.getenv(
        "HELPLINE_REFERRAL_TEXT", "Please contact a qualified clinician or your local public-health service."
    ).strip()
    EMERGENCY_NUMBER: str = os.getenv("EMERGENCY_NUMBER", "112").strip()
    MENTAL_HEALTH_HELPLINE: str = os.getenv("MENTAL_HEALTH_HELPLINE", "14416").strip()
    WEB_ALLOWED_HOSTS: str = os.getenv(
        "WEB_ALLOWED_HOSTS", "who.int,mohfw.gov.in,icmr.gov.in,nhp.gov.in,nhm.gov.in,cdc.gov,nhs.uk,medlineplus.gov,nih.gov"
    ).strip()
    WEB_CACHE_TTL_SECONDS: int = int(os.getenv("WEB_CACHE_TTL_SECONDS", "3600"))
    ENABLE_CONTENT_VERIFIER: bool = os.getenv("ENABLE_CONTENT_VERIFIER", "true").strip().lower() in {"1", "true", "yes"}
    DATABASE_URL: str = os.getenv("DATABASE_URL", "").strip()

    @property
    def parsed_origins(self) -> list[str]:
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def parsed_web_hosts(self) -> list[str]:
        return [host.strip().lower() for host in self.WEB_ALLOWED_HOSTS.split(",") if host.strip()]
settings = Settings()

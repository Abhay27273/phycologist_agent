from typing import List, Literal, Union, Any
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator

# Find the project root (where .env lives)
# This file is at: psych-platform-core/app/core/config.py
# So we go up 2 levels to reach psych-platform-core/
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = BASE_DIR / ".env"

class Settings(BaseSettings):
    """
    Application Configuration.
    Validates all environment variables on startup.
    """
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding='utf-8',
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False
    )

    # Application
    APP_NAME: str = "Psych-Platform-Core"
    ENVIRONMENT: Literal["local", "dev", "prod"] = "local"
    LOG_LEVEL: str = "INFO"
    API_PREFIX: str = "/api/v1"
    
    # Security & CORS
    # We allow str initially to handle the parsing from .env
    CORS_ORIGINS: str = "http://localhost:3000"

    # JWT Authentication
    JWT_SECRET_KEY: str = Field(default="change-me-in-production", description="JWT signing secret")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Redis (rate limiting + RAG cache)
    REDIS_URL: str = Field(default="memory://", description="Redis URL for rate limiting; falls back to in-memory")
    
    # Infrastructure (AWS/Database)
    DATABASE_URL: str = Field(..., description="Database Connection String")
    
    # AI Services
    GOOGLE_API_KEY: str = Field(..., description="Google Gemini API Key")
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Groq (free-tier fallback)
    GROQ_API_KEY: str = Field(default="", description="Groq API Key")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # RAG / Vector DB — backend selector
    VECTOR_DB_BACKEND: Literal["pinecone", "qdrant"] = "pinecone"

    # Pinecone (optional when using qdrant backend)
    PINECONE_API_KEY: str = Field(default="", description="Pinecone Vector DB Key")
    PINECONE_INDEX_NAME: str = "psych-brain"

    # Qdrant — local file-based (default) or server mode (Docker / multi-worker)
    QDRANT_COLLECTION_NAME: str = "psych-brain"
    QDRANT_PATH: str = "./qdrant_data"
    QDRANT_MODE: str = "local"   # "local" | "server"
    QDRANT_URL: str = "http://127.0.0.1:6333"  # used when QDRANT_MODE=server

    # Real-time voice (Phase V) — Deepgram streaming STT + TTS
    DEEPGRAM_API_KEY: str = Field(default="", description="Deepgram API key; voice endpoints 503 if unset")
    DEEPGRAM_STT_MODEL: str = "nova-3"
    DEEPGRAM_TTS_MODEL: str = "aura-2-thalia-en"
    VOICE_ENABLED: bool = True

    # --- VALIDATORS ---

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v: Any) -> str:
        """
        Ensures CORS_ORIGINS is a string.
        The server.py will split it when needed.
        """
        if isinstance(v, (list, tuple)):
            return ",".join(v)
        return v

    @model_validator(mode="after")
    def validate_backend_credentials(self) -> "Settings":
        if self.VECTOR_DB_BACKEND == "pinecone" and not self.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required when VECTOR_DB_BACKEND=pinecone")
        return self

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def force_ip_on_windows(cls, v: str) -> str:
        """
        Fixes [Errno 11003] on Windows by forcing 127.0.0.1
        instead of localhost for asyncpg connections.
        """
        if v and "localhost" in v:
            return v.replace("localhost", "127.0.0.1")
        return v

settings = Settings()
"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration parameters."""

    APP_NAME: str = "RAG Hallucination Evaluator"
    DEBUG: bool = False
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # LLM Provider credentials & endpoints
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    HF_MODEL_ID: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Default runtime choices
    DEFAULT_LLM_BACKEND: str = "groq"
    DEFAULT_GROQ_MODEL: str = "llama3-8b-8192"

    # Models
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Storage paths
    FAISS_INDEX_PATH: str = "data/faiss_index"
    UPLOADS_PATH: str = "data/uploads"
    EXPERIMENTS_PATH: str = "data/experiments"
    DATABASE_URL: str = "sqlite:///./data/research.db"

    # RAG defaults
    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 128
    DEFAULT_TOP_K: int = 5
    MAX_FILE_SIZE_MB: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()

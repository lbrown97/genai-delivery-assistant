from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized environment-backed runtime settings."""

    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "qwen3.5:9b"

    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION: str = "delivery_docs"

    LANGFUSE_HOST: str = "http://langfuse:3000"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""

    APP_ENV: str = "local"


settings = Settings()

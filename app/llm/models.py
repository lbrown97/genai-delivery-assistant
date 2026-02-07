from langchain_ollama import ChatOllama

from app.core.settings import settings


def get_chat_model(temperature: float = 0.2):
    """Return the configured Ollama chat model client."""

    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=temperature,
    )

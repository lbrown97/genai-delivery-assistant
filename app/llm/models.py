from langchain_ollama import ChatOllama

from app.core.settings import settings


def get_chat_model(
    temperature: float = 0.2,
    *,
    reasoning: bool = False,
    format: str | None = None,
):
    """Return the configured Ollama chat model client."""

    kwargs = {
        "base_url": settings.OLLAMA_BASE_URL,
        "model": settings.OLLAMA_MODEL,
        "temperature": temperature,
        "reasoning": reasoning,
    }
    if format is not None:
        kwargs["format"] = format

    return ChatOllama(
        **kwargs,
    )

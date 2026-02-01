import logging
import os
from typing import Any, Optional

log = logging.getLogger(__name__)


def get_langfuse_handler() -> Optional[Any]:
    """
    Returns Langfuse LangChain CallbackHandler if configured, else None.
    Keeps API functional even if Langfuse is not running.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")

    if not (public_key and secret_key and host):
        return None

    try:
        # Langfuse v3: CallbackHandler reads auth/host from env.
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as e:
        log.warning("Langfuse disabled (import/init failed): %s", e)
        return None

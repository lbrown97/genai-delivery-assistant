import os

from langchain_huggingface import HuggingFaceEmbeddings


def _pick_device() -> str:
    """Select embedding device from env override or CUDA availability."""

    env = os.getenv("EMBEDDINGS_DEVICE", "auto").strip().lower()
    if env and env != "auto":
        return env
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_embedding_model():
    """Return the configured embedding model instance."""

    # English-optimized embeddings
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": _pick_device()},
    )

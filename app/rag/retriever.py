import contextvars
import os
import re

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.settings import settings
from app.rag.embeddings import get_embedding_model


def get_vectorstore():
    client = QdrantClient(url=settings.QDRANT_URL)
    embeddings = get_embedding_model()
    if not client.collection_exists(collection_name=settings.QDRANT_COLLECTION):
        dim = len(embeddings.embed_query("dimension check"))
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION,
        embedding=embeddings,
    )


def _mode() -> str:
    override = _retrieval_mode_override.get()
    if override:
        return override.strip().lower()
    return os.getenv("RETRIEVAL_MODE", "mmr").strip().lower()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def get_retriever(k: int = 6):
    vs = get_vectorstore()
    mode = _mode()
    if mode == "mmr":
        fetch_k = int(os.getenv("MMR_FETCH_K", "24"))
        lambda_mult = float(os.getenv("MMR_LAMBDA", "0.5"))
        return vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )
    return vs.as_retriever(search_kwargs={"k": k})


def get_retriever_with_scores(query: str, k: int = 6):
    vs = get_vectorstore()
    mode = _mode()
    if mode == "mmr":
        fetch_k = int(os.getenv("MMR_FETCH_K", "24"))
        lambda_mult = float(os.getenv("MMR_LAMBDA", "0.5"))
        mmr_docs = vs.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
        scored = vs.similarity_search_with_score(query, k=fetch_k)
        score_map = {d.page_content: s for d, s in scored}
        return [(d, score_map.get(d.page_content, 1.0)) for d in mmr_docs]

    if mode == "hybrid":
        fetch_k = int(os.getenv("HYBRID_FETCH_K", "24"))
        alpha = float(os.getenv("HYBRID_ALPHA", "0.7"))
        scored = vs.similarity_search_with_score(query, k=fetch_k)
        q_tokens = _tokenize(query)
        ranked = []
        for d, dist in scored:
            d_tokens = _tokenize(d.page_content)
            overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
            sim = 1.0 / (1.0 + float(dist))
            combined = alpha * sim + (1.0 - alpha) * overlap
            ranked.append((d, 1.0 - combined))
        ranked.sort(key=lambda x: x[1])
        return ranked[:k]

    return vs.similarity_search_with_score(query, k=k)


_retrieval_mode_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "retrieval_mode_override", default=None
)


def set_retrieval_mode_override(mode: str | None):
    _retrieval_mode_override.set(mode)


def clear_retrieval_mode_override():
    _retrieval_mode_override.set(None)

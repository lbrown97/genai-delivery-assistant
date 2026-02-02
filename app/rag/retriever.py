import contextvars
import os
import re

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

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


def _doc_scope() -> str:
    override = _doc_scope_override.get()
    if override:
        return override.strip().lower()
    return os.getenv("DOC_SCOPE", "project").strip().lower()


def _allowed_doc_types() -> set[str] | None:
    scope = _doc_scope()
    if scope in {"all", "*"}:
        return None
    if scope == "external":
        return {"external"}
    return {"project"}


def _qdrant_filter():
    allowed = _allowed_doc_types()
    if not allowed:
        return None
    value = next(iter(allowed))
    return Filter(must=[FieldCondition(key="metadata.doc_type", match=MatchValue(value=value))])


def _similarity_with_score(vs: QdrantVectorStore, query: str, k: int, q_filter):
    if q_filter is not None:
        try:
            return vs.similarity_search_with_score(query, k=k, filter=q_filter)
        except TypeError:
            pass
    return vs.similarity_search_with_score(query, k=k)


def _filter_by_doc_type(pairs: list[tuple], allowed: set[str] | None):
    if allowed is None:
        return pairs
    return [(d, s) for d, s in pairs if d.metadata.get("doc_type", "project") in allowed]


def _filter_docs_by_type(docs: list, allowed: set[str] | None):
    if allowed is None:
        return docs
    return [d for d in docs if d.metadata.get("doc_type", "project") in allowed]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def get_retriever_with_scores(query: str, k: int = 6):
    vs = get_vectorstore()
    mode = _mode()
    allowed = _allowed_doc_types()
    q_filter = _qdrant_filter()
    if mode == "mmr":
        fetch_k = int(os.getenv("MMR_FETCH_K", "24"))
        lambda_mult = float(os.getenv("MMR_LAMBDA", "0.5"))
        mmr_docs = vs.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
        scored = _similarity_with_score(vs, query, k=fetch_k, q_filter=q_filter)
        scored = _filter_by_doc_type(scored, allowed)
        mmr_docs = _filter_docs_by_type(mmr_docs, allowed)
        score_map = {d.page_content: s for d, s in scored}
        return [(d, score_map.get(d.page_content, 1.0)) for d in mmr_docs]

    if mode == "hybrid":
        fetch_k = int(os.getenv("HYBRID_FETCH_K", "24"))
        alpha = float(os.getenv("HYBRID_ALPHA", "0.7"))
        scored = _similarity_with_score(vs, query, k=fetch_k, q_filter=q_filter)
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

    return _similarity_with_score(vs, query, k=k, q_filter=q_filter)


_retrieval_mode_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "retrieval_mode_override", default=None
)


def set_retrieval_mode_override(mode: str | None):
    _retrieval_mode_override.set(mode)


def clear_retrieval_mode_override():
    _retrieval_mode_override.set(None)


_doc_scope_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "doc_scope_override", default=None
)


def set_doc_scope_override(scope: str | None):
    _doc_scope_override.set(scope)


def clear_doc_scope_override():
    _doc_scope_override.set(None)

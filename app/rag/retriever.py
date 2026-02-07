import contextvars
import os
import re

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from app.core.env import env_float, env_int
from app.core.settings import settings
from app.rag.embeddings import get_embedding_model

_VECTORSTORE: QdrantVectorStore | None = None


def get_vectorstore():
    """Return a cached Qdrant vector store instance, creating collection if needed."""

    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    client = QdrantClient(url=settings.QDRANT_URL)
    embeddings = get_embedding_model()
    if not client.collection_exists(collection_name=settings.QDRANT_COLLECTION):
        dim = len(embeddings.embed_query("dimension check"))
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    _VECTORSTORE = QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION,
        embedding=embeddings,
    )
    return _VECTORSTORE


def _mode() -> str:
    """Resolve retrieval mode from request override or environment default."""

    override = _retrieval_mode_override.get()
    if override:
        return override.strip().lower()
    return os.getenv("RETRIEVAL_MODE", "auto").strip().lower()


def _doc_scope() -> str:
    """Resolve document scope from request override or environment default."""

    override = _doc_scope_override.get()
    if override:
        return override.strip().lower()
    return os.getenv("DOC_SCOPE", "project").strip().lower()


def _k(request_k: int | None) -> int:
    """Resolve top-k with precedence: request override > explicit arg > env default."""

    override = _retrieval_k_override.get()
    if override is not None:
        return max(1, int(override))
    if request_k is not None:
        return max(1, int(request_k))
    env_k = env_int("RETRIEVAL_K", 6)
    return max(1, env_k)


def _doc_type_scope() -> str:
    """Normalize scope into one of: all, external, project."""

    scope = _doc_scope()
    if scope in {"all", "*"}:
        return "all"
    if scope in {"external", "project"}:
        return scope
    return "project"


def get_active_doc_scope() -> str:
    """Expose the currently effective retrieval document scope."""

    return _doc_type_scope()


def _qdrant_filter():
    """Build a Qdrant payload filter for document scope when required."""

    scope = _doc_type_scope()
    if scope == "all":
        return None
    return Filter(
        must=[
            FieldCondition(
                key="metadata.doc_type",
                match=MatchValue(value=scope),
            )
        ]
    )


def _similarity_with_score(vs: QdrantVectorStore, query: str, k: int, q_filter):
    """Run similarity search with scores, applying filter when supported."""

    if q_filter is not None:
        try:
            return vs.similarity_search_with_score(query, k=k, filter=q_filter)
        except TypeError:
            pass
    return vs.similarity_search_with_score(query, k=k)


def _tokenize(text: str) -> set[str]:
    """Tokenize text to alphanumeric lowercase terms for lexical overlap."""

    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def _doc_key(doc):
    """Build a stable key for matching documents across retrieval result sets."""

    meta = getattr(doc, "metadata", {}) or {}
    return (
        meta.get("_id"),
        meta.get("source_id"),
        meta.get("h1"),
        getattr(doc, "page_content", None),
    )


def get_retriever_with_scores(query: str, k: int | None = None):
    """Retrieve `(document, score)` pairs using selected retrieval strategy."""

    vs = get_vectorstore()
    mode = _mode()
    scope = _doc_type_scope()
    if mode == "auto":
        mode = "hybrid" if scope == "external" else "mmr"
    k = _k(k)
    q_filter = _qdrant_filter()
    if mode == "mmr":
        fetch_k = max(k, env_int("MMR_FETCH_K", 24))
        lambda_mult = env_float("MMR_LAMBDA", 0.5)
        mmr_docs = vs.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=q_filter,
        )
        scored = _similarity_with_score(vs, query, k=fetch_k, q_filter=q_filter)
        score_map = {_doc_key(d): s for d, s in scored}
        fallback = max((float(s) for _, s in scored), default=1.0)
        mmr_pairs = [(d, score_map.get(_doc_key(d), fallback)) for d in mmr_docs]
        return mmr_pairs[:k]

    if mode == "hybrid":
        base_fetch_k = env_int("HYBRID_FETCH_K", 24)
        if scope == "external":
            external_min = env_int("HYBRID_FETCH_K_EXTERNAL_MIN", 60)
            external_multiplier = env_int("HYBRID_FETCH_K_EXTERNAL_MULTIPLIER", 5)
            external_max = env_int("HYBRID_FETCH_K_EXTERNAL_MAX", 120)
            dynamic_target = max(base_fetch_k, external_min, k * max(1, external_multiplier))
            fetch_k = max(k, min(max(k, external_max), dynamic_target))
        else:
            fetch_k = max(k, base_fetch_k)
        if scope == "external":
            alpha = env_float("HYBRID_ALPHA_EXTERNAL", env_float("HYBRID_ALPHA", 0.7))
        elif scope == "project":
            alpha = env_float("HYBRID_ALPHA_PROJECT", env_float("HYBRID_ALPHA", 0.7))
        else:
            alpha = env_float("HYBRID_ALPHA_ALL", env_float("HYBRID_ALPHA", 0.7))
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

    fetch_k = max(k, env_int("SIMILARITY_FETCH_K", 24))
    scored = _similarity_with_score(vs, query, k=fetch_k, q_filter=q_filter)
    return scored[:k]


_retrieval_mode_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "retrieval_mode_override", default=None
)


def set_retrieval_mode_override(mode: str | None):
    """Set request-scoped retrieval mode override."""

    _retrieval_mode_override.set(mode)


def clear_retrieval_mode_override():
    """Clear request-scoped retrieval mode override."""

    _retrieval_mode_override.set(None)


_doc_scope_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "doc_scope_override", default=None
)


def set_doc_scope_override(scope: str | None):
    """Set request-scoped document-scope override."""

    _doc_scope_override.set(scope)


def clear_doc_scope_override():
    """Clear request-scoped document-scope override."""

    _doc_scope_override.set(None)


_retrieval_k_override: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "retrieval_k_override", default=None
)


def set_retrieval_k_override(k: int | None):
    """Set request-scoped top-k override."""

    _retrieval_k_override.set(None if k is None else max(1, int(k)))


def clear_retrieval_k_override():
    """Clear request-scoped top-k override."""

    _retrieval_k_override.set(None)

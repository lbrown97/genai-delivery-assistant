import contextvars
import os
import re

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from app.core.settings import settings
from app.rag.embeddings import get_embedding_model

_VECTORSTORE: QdrantVectorStore | None = None


def get_vectorstore():
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
    override = _retrieval_mode_override.get()
    if override:
        return override.strip().lower()
    return os.getenv("RETRIEVAL_MODE", "mmr").strip().lower()


def _doc_scope() -> str:
    override = _doc_scope_override.get()
    if override:
        return override.strip().lower()
    return os.getenv("DOC_SCOPE", "project").strip().lower()


def _k(default: int) -> int:
    override = _retrieval_k_override.get()
    if override is not None:
        return max(1, int(override))
    env_k = _env_int("RETRIEVAL_K", default)
    if env_k > 0:
        return max(1, env_k)
    return max(1, int(default))


def _doc_type_scope() -> str:
    scope = _doc_scope()
    if scope in {"all", "*"}:
        return "all"
    if scope in {"external", "project"}:
        return scope
    return "project"


def _qdrant_filter():
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
    if q_filter is not None:
        try:
            return vs.similarity_search_with_score(query, k=k, filter=q_filter)
        except TypeError:
            pass
    return vs.similarity_search_with_score(query, k=k)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def _identifier_tokens(text: str) -> set[str]:
    # Keep identifier boosting narrow to avoid false positives like "Top 10".
    return {m.upper() for m in re.findall(r"\bllm\d{1,3}\b", text.lower())}


def _effective_fetch_k(base_k: int, identifier_tokens: set[str]) -> int:
    fetch_k = base_k
    if identifier_tokens:
        fetch_k = max(fetch_k, _env_int("IDENTIFIER_FETCH_K", 500))
    return fetch_k


def _doc_text(doc) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    source_id = str(meta.get("source_id", ""))
    h1 = str(meta.get("h1", ""))
    content = getattr(doc, "page_content", "") or ""
    return f"{source_id}\n{h1}\n{content}".upper()


def _apply_identifier_boost(
    pairs: list[tuple],
    identifier_tokens: set[str],
) -> list[tuple]:
    if not identifier_tokens:
        return pairs

    boost_per_hit = _env_float("IDENTIFIER_BOOST", 0.2)
    hit_pairs = []
    miss_pairs = []
    for d, s in pairs:
        score = float(s)
        text = _doc_text(d)
        hits = sum(1 for token in identifier_tokens if token in text)
        boosted_score = max(0.0, score - hits * boost_per_hit)
        pair = (d, boosted_score)
        if hits > 0:
            hit_pairs.append(pair)
        else:
            miss_pairs.append(pair)
    hit_pairs.sort(key=lambda x: x[1])
    miss_pairs.sort(key=lambda x: x[1])
    return hit_pairs + miss_pairs


def _doc_key(doc):
    meta = getattr(doc, "metadata", {}) or {}
    return (
        meta.get("_id"),
        meta.get("source_id"),
        meta.get("h1"),
        getattr(doc, "page_content", None),
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def get_retriever_with_scores(query: str, k: int = 6):
    vs = get_vectorstore()
    mode = _mode()
    k = _k(k)
    q_filter = _qdrant_filter()
    identifier_tokens = _identifier_tokens(query)
    if mode == "mmr":
        fetch_k = _effective_fetch_k(
            max(k, _env_int("MMR_FETCH_K", 24)),
            identifier_tokens,
        )
        lambda_mult = _env_float("MMR_LAMBDA", 0.5)
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
        return _apply_identifier_boost(mmr_pairs, identifier_tokens)[:k]

    if mode == "hybrid":
        fetch_k = _effective_fetch_k(
            max(k, _env_int("HYBRID_FETCH_K", 24)),
            identifier_tokens,
        )
        alpha = _env_float("HYBRID_ALPHA", 0.7)
        scored = _similarity_with_score(vs, query, k=fetch_k, q_filter=q_filter)
        q_tokens = _tokenize(query)
        ranked = []
        for d, dist in scored:
            d_tokens = _tokenize(d.page_content)
            overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
            sim = 1.0 / (1.0 + float(dist))
            combined = alpha * sim + (1.0 - alpha) * overlap
            ranked.append((d, 1.0 - combined))
        boosted = _apply_identifier_boost(ranked, identifier_tokens)
        return boosted[:k]

    fetch_k = _effective_fetch_k(
        max(k, _env_int("SIMILARITY_FETCH_K", 24)),
        identifier_tokens,
    )
    scored = _similarity_with_score(vs, query, k=fetch_k, q_filter=q_filter)
    boosted = _apply_identifier_boost(scored, identifier_tokens)
    return boosted[:k]


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


_retrieval_k_override: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "retrieval_k_override", default=None
)


def set_retrieval_k_override(k: int | None):
    _retrieval_k_override.set(None if k is None else max(1, int(k)))


def clear_retrieval_k_override():
    _retrieval_k_override.set(None)

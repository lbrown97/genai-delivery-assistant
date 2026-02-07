from app.rag.citations import collect_sources, format_context
from app.rag.retriever import get_retriever_with_scores


def retrieve_context_with_scores(query: str, k: int | None = None):
    """Return retrieved docs, scores, formatted context, and normalized sources."""

    pairs = get_retriever_with_scores(query, k=k)
    docs = [d for d, _ in pairs]
    scores = [s for _, s in pairs]
    return {
        "docs": docs,
        "scores": scores,
        "context": format_context(docs),
        "sources": collect_sources(docs),
    }

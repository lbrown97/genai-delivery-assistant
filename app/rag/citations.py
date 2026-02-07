from typing import Any, Dict, List


def format_context(docs) -> str:
    """Format retrieved documents into a prompt-ready context string."""

    # docs are LangChain Documents
    parts: List[str] = []
    for d in docs:
        sid = d.metadata.get("source_id", "unknown_source")
        parts.append(f"[{sid}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def collect_sources(docs) -> List[Dict[str, Any]]:
    """Collect unique source metadata from retrieved documents."""

    out = []
    for d in docs:
        out.append(
            {
                "source_id": d.metadata.get("source_id", "unknown_source"),
                "meta": {k: v for k, v in d.metadata.items() if k != "text"},
            }
        )
    # unique by source_id
    uniq = {x["source_id"]: x for x in out}
    return list(uniq.values())

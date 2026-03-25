import os
from collections import defaultdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document

from app.agent.guardrails import redact_pii
from app.rag.chunking import split_documents
from app.rag.retriever import get_vectorstore

SUPPORTED = {".md", ".txt", ".pdf"}
SCROLL_BATCH_SIZE = 256
DELETE_BATCH_SIZE = 128


def _chunk_point_id(source_id: str, chunk_index: int) -> str:
    """Build a deterministic point ID for a chunk within a source document."""

    return str(uuid5(NAMESPACE_URL, f"{source_id}::{chunk_index}"))


def _matches_replaced_source(
    source_id: str | None,
    source_path: str | None,
    target_paths: set[str],
) -> bool:
    """Return whether a stored point belongs to a source path being replaced."""

    if source_path and source_path in target_paths:
        return True
    if not source_id:
        return False
    if source_id in target_paths:
        return True
    return any(source_id.startswith(f"{path}#p") for path in target_paths)


def _prepare_chunks_for_upsert(chunks: list[Document]) -> set[str]:
    """Attach deterministic IDs/indices to chunks and return source paths to replace."""

    counters: defaultdict[str, int] = defaultdict(int)
    source_paths: set[str] = set()

    for chunk in chunks:
        source_id = chunk.metadata.get("source_id", "unknown_source")
        source_path = chunk.metadata.get("source_path") or source_id.split("#p", 1)[0]
        chunk_index = counters[source_id]
        counters[source_id] += 1

        chunk.metadata["source_path"] = source_path
        chunk.metadata["chunk_index"] = chunk_index
        chunk.id = _chunk_point_id(source_id, chunk_index)
        source_paths.add(source_path)

    return source_paths


def _delete_existing_points_for_sources(vs, source_paths: set[str]) -> int:
    """Delete existing points for the ingested source paths before re-adding them."""

    if not source_paths:
        return 0

    client = vs.client
    collection = vs.collection_name
    offset = None
    point_ids: list[str | int] = []

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            if _matches_replaced_source(
                metadata.get("source_id"),
                metadata.get("source_path"),
                source_paths,
            ):
                point_ids.append(point.id)
        if offset is None:
            break

    for i in range(0, len(point_ids), DELETE_BATCH_SIZE):
        client.delete(
            collection_name=collection,
            points_selector=point_ids[i : i + DELETE_BATCH_SIZE],
            wait=True,
        )

    return len(point_ids)


def load_documents(data_dir: str) -> list[Document]:
    """Load supported files from disk into LangChain documents with metadata."""

    docs = []
    base = Path(data_dir)
    paths = sorted(base.rglob("*"), key=lambda path: path.relative_to(base).as_posix())
    for p in paths:
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            rel = p.relative_to(base).as_posix()
            doc_type = "external" if rel.startswith("external/") else "project"
            if p.suffix.lower() == ".pdf":
                try:
                    from pypdf import PdfReader

                    reader = PdfReader(str(p))
                    for i, page in enumerate(reader.pages, start=1):
                        text = page.extract_text() or ""
                        text = redact_pii(text)
                        if text.strip():
                            docs.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source_id": f"{rel}#p{i}",
                                        "source_path": rel,
                                        "doc_type": doc_type,
                                    },
                                )
                            )
                except Exception:
                    continue
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")
                text = redact_pii(text)
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source_id": rel,
                            "source_path": rel,
                            "doc_type": doc_type,
                        },
                    )
                )
    return docs


def ingest(data_dir: str = "data"):
    """Ingest loaded documents into Qdrant after chunking."""

    raw_docs = load_documents(data_dir)
    if not raw_docs:
        return {"status": "no_docs_found", "count": 0}

    chunks = split_documents(raw_docs)
    source_paths = _prepare_chunks_for_upsert(chunks)

    vs = get_vectorstore()
    replaced_points = _delete_existing_points_for_sources(vs, source_paths)
    vs.add_documents(chunks)

    return {
        "status": "ok",
        "raw_docs": len(raw_docs),
        "chunks": len(chunks),
        "replaced_sources": len(source_paths),
        "replaced_points": replaced_points,
    }


if __name__ == "__main__":
    print(ingest(os.getenv("DATA_DIR", "data")))

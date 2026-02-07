import os
from pathlib import Path

from langchain_core.documents import Document

from app.agent.guardrails import redact_pii
from app.rag.chunking import split_documents
from app.rag.retriever import get_vectorstore

SUPPORTED = {".md", ".txt", ".pdf"}


def load_documents(data_dir: str) -> list[Document]:
    """Load supported files from disk into LangChain documents with metadata."""

    docs = []
    base = Path(data_dir)
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            rel = str(p.relative_to(base))
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
                        metadata={"source_id": rel, "doc_type": doc_type},
                    )
                )
    return docs


def ingest(data_dir: str = "data"):
    """Ingest loaded documents into Qdrant after chunking."""

    raw_docs = load_documents(data_dir)
    if not raw_docs:
        return {"status": "no_docs_found", "count": 0}

    chunks = split_documents(raw_docs)

    vs = get_vectorstore()
    vs.add_documents(chunks)

    return {"status": "ok", "raw_docs": len(raw_docs), "chunks": len(chunks)}


if __name__ == "__main__":
    print(ingest(os.getenv("DATA_DIR", "data")))

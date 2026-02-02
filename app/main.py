import os

import requests
from fastapi import FastAPI, Request
from langfuse import Langfuse
from pydantic import BaseModel
from qdrant_client import QdrantClient

from app.agent.router import agent_route
from app.core.logging import setup_logging
from app.core.settings import settings
from app.rag.ingest import ingest
from app.rag.retriever import (
    clear_doc_scope_override,
    clear_retrieval_mode_override,
    get_retriever_with_scores,
    set_doc_scope_override,
    set_retrieval_mode_override,
)

setup_logging()
app = FastAPI(title="GenAI Delivery Assistant")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


def _check_required_env():
    required = ["OLLAMA_BASE_URL", "QDRANT_URL"]
    missing = [k for k in required if not os.getenv(k)]
    return missing


@app.get("/ready")
def ready():
    missing = _check_required_env()
    checks = {"env_missing": missing}

    ollama = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
    qdrant = os.getenv("QDRANT_URL", "").rstrip("/")

    def _ping(url: str, path: str):
        if not url:
            return False
        try:
            r = requests.get(f"{url}{path}", timeout=2)
            return r.status_code < 500
        except Exception:
            return False

    checks["ollama"] = _ping(ollama, "/api/tags")
    checks["qdrant"] = _ping(qdrant, "/collections")

    ready_ok = not missing and checks["ollama"] and checks["qdrant"]
    return {"ready": ready_ok, "checks": checks}


@app.post("/ingest")
def ingest_docs():
    return ingest("data")


@app.post("/agent")
def agent(req: AskRequest, request: Request):
    mode = request.headers.get("X-Retrieval-Mode")
    scope = request.headers.get("X-Doc-Scope")
    if mode:
        set_retrieval_mode_override(mode)
    if scope:
        set_doc_scope_override(scope)
    try:
        return agent_route(req.question)
    finally:
        if mode:
            clear_retrieval_mode_override()
        if scope:
            clear_doc_scope_override()


@app.get("/debug/langfuse")
def debug_langfuse():
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")

    if not (pk and sk and host):
        return {"enabled": False, "reason": "Missing LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST"}

    lf = Langfuse(public_key=pk, secret_key=sk, host=host)
    return {"enabled": True, "auth_check": lf.auth_check()}


@app.get("/debug/qdrant")
def debug_qdrant(limit: int = 10):
    client = QdrantClient(url=settings.QDRANT_URL)
    collection = settings.QDRANT_COLLECTION
    if not client.collection_exists(collection_name=collection):
        return {"collection": collection, "exists": False}

    info = client.get_collection(collection_name=collection)
    points, next_offset = client.scroll(
        collection_name=collection,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    samples = []
    for p in points:
        payload = p.payload or {}
        meta = payload.get("metadata") or {}
        samples.append(
            {
                "id": p.id,
                "source_id": meta.get("source_id"),
                "doc_type": meta.get("doc_type"),
                "metadata_keys": sorted(payload.keys()),
            }
        )

    return {
        "collection": collection,
        "exists": True,
        "points_count": getattr(info, "points_count", None),
        "samples": samples,
        "next_offset": next_offset,
    }


@app.get("/debug/retrieval")
def debug_retrieval(query: str, k: int = 6, mode: str | None = None, scope: str | None = None):
    if mode:
        set_retrieval_mode_override(mode)
    if scope:
        set_doc_scope_override(scope)
    try:
        pairs = get_retriever_with_scores(query, k=k)
    finally:
        if mode:
            clear_retrieval_mode_override()
        if scope:
            clear_doc_scope_override()

    out = []
    for d, s in pairs:
        meta = d.metadata or {}
        out.append(
            {
                "score": s,
                "source_id": meta.get("source_id"),
                "doc_type": meta.get("doc_type"),
                "h1": meta.get("h1"),
            }
        )
    return {"query": query, "k": k, "results": out}

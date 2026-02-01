from fastapi import FastAPI, Request
import os
import requests
from pydantic import BaseModel
from app.core.logging import setup_logging
from app.rag.ingest import ingest
from app.rag.retriever import set_retrieval_mode_override, clear_retrieval_mode_override
from app.agent.router import agent_route

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
    if mode:
        set_retrieval_mode_override(mode)
    try:
        return agent_route(req.question)
    finally:
        if mode:
            clear_retrieval_mode_override()

@app.get("/debug/langfuse")
def debug_langfuse():
    import os
    from langfuse import Langfuse

    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")

    if not (pk and sk and host):
        return {"enabled": False, "reason": "Missing LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST"}

    lf = Langfuse(public_key=pk, secret_key=sk, host=host)
    return {"enabled": True, "auth_check": lf.auth_check()}

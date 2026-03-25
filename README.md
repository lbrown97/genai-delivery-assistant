GenAI Delivery Assistant
========================

Agentic, retrieval-augmented assistant for consulting delivery. It answers questions
from project artifacts and produces structured outputs (ADR, solution outline, user
stories, risk assessment) with citations and guardrails.

Why this project
----------------
- End-to-end RAG with local LLM (Ollama) and vector DB (Qdrant)
- Agentic tool routing with structured outputs
- Guardrails (schema validation + PII redaction + citation enforcement)
- Evaluation (RAGAS + router tool accuracy)
- Observability with Langfuse

Key features
------------
- Ingestion: PDF / Markdown / TXT
- Chunking: Markdown-aware + recursive splitting
- Retrieval: similarity, MMR (default), or hybrid
- Answers with citations or refusal
- Structured JSON outputs (ADR, solution outline, user stories, risk assessment)
- PII redaction (best-effort baseline)
- Langfuse traces and tags
- RAGAS and router evaluation

Architecture
------------
```mermaid
flowchart LR
  UI[Streamlit UI] --> API[FastAPI /agent]
  API --> Router[Agent Router]
  Router --> Retriever[Retriever]
  Retriever --> Qdrant[(Qdrant)]
  Router --> Ollama[(Ollama)]
  API --> Langfuse[(Langfuse)]
  Data[(Artifacts: PDFs/MD/TXT)] --> Ingest[Ingestion + Chunking + Embeddings]
  Ingest --> Qdrant
```

See docs for more: `docs/architecture.md`

Quickstart
----------
1) Copy environment template:
   ```bash
   cp .env.example .env
   ```
2) Start core services:
   ```bash
   docker compose up -d
   ```
3) Download external reference PDFs (optional, needed for external-question demos):
   ```bash
   make external-data
   ```
4) Ingest data (API or UI):
   ```bash
   curl -X POST http://localhost:8000/ingest
   ```
   - Or, after opening the UI, click `Ingest /data` in the sidebar.
   - Re-ingesting the same source files replaces their existing chunks instead of duplicating them.
5) Open UI:
   - http://localhost:8501
6) Try sample questions:
   - `docs/sample_questions.md`

Data layout
-----------
- `data/` is ingested recursively.
- Files under `data/external/` are tagged as `doc_type=external`.
- Files under the rest of `data/` are tagged as `doc_type=project`.
- External PDFs are intentionally not tracked in git; see `data/external/README.md`.

Observability (optional)
------------------------
Start Langfuse stack:
```bash
docker compose --profile obs up -d
```
Langfuse UI: http://localhost:3000

Evaluation
----------
- RAGAS:
  ```bash
  make eval
  ```
  Output: `app/eval/results.json`
- Router accuracy:
  ```bash
  make eval-router
  ```
  Output: `app/eval/router_eval_results.json`

Retrieval modes
---------------
- Default: `RETRIEVAL_MODE=auto` (uses `mmr` for `project` scope and `hybrid` for `external` scope)
- Override per request using UI dropdown or header:
  - Headers:
    - `X-Retrieval-Mode: mmr|hybrid|similarity`
    - `X-Doc-Scope: project|external|all`
    - `X-Retrieval-K: <int>`
- Env options:
  - `RETRIEVAL_MODE=auto|mmr|hybrid|similarity`
  - `RETRIEVAL_K=6` (default top-k when no header override is sent)
  - `DOC_SCOPE=project|external|all`
  - `MMR_FETCH_K=24`, `MMR_LAMBDA=0.5`
  - `HYBRID_FETCH_K=24`, `HYBRID_FETCH_K_EXTERNAL_MIN=60`, `HYBRID_FETCH_K_EXTERNAL_MULTIPLIER=5`, `HYBRID_FETCH_K_EXTERNAL_MAX=120`, `HYBRID_ALPHA=0.7`
  - `HYBRID_ALPHA_PROJECT=0.7`, `HYBRID_ALPHA_EXTERNAL=0.45`, `HYBRID_ALPHA_ALL=0.65`
  - `SIMILARITY_FETCH_K=24`
- Note: `*_FETCH_K*` controls retrieval candidate depth only; prompt context still uses returned top-`k`.

Guardrails and safety
---------------------
- Schema validation with GuardrailsAI (fallback to Pydantic)
- PII redaction (regex-based; best-effort)
- Refuse if missing/invalid citations or insufficient context

Security note
-------------
- Included artifacts are synthetic. Do not ingest real customer data without stronger production controls.
- PII redaction is best-effort and may miss edge cases.

Health checks
-------------
- Liveness: `GET /health`
- Readiness: `GET /ready` (checks env + Ollama + Qdrant)

Developer tools
---------------
- Lint: `make lint`
- Tests (container): `docker compose exec api pytest -q`
- Tests (host, if deps installed): `make test`
- Debug API with VS Code: `make debug` and attach to port 5678

Debug endpoints
---------------
- `GET /debug/retrieval` (inspect retrieved chunks/scores for a query)
- `GET /debug/qdrant` (inspect collection samples and payload keys)
- `GET /debug/langfuse` (check SDK auth wiring)

Examples, FAQ, and Test Prompts
-------------------------------
- Examples: `docs/examples/`
- Project FAQ: `docs/project_faq.md`
- Sample questions for testing: `docs/sample_questions.md`

Notes
-----
- Changing embeddings requires re-ingest.
- PDF extraction quality depends on PDF text extractability.

License
-------
MIT. See `LICENSE`.

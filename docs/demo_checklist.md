Demo Checklist
==============

Setup
-----
- Start services: `docker compose up -d`
- Optional tracing: `docker compose --profile obs up -d`
- Ingest data: `curl -X POST http://localhost:8000/ingest`
- Open UI: http://localhost:8501

Core demo flow
--------------
1) Ask a general question (RAG + citations).
2) Show the selected tool in the response.
3) Use structured inputs (toggle in UI).
4) Switch retrieval mode (MMR vs hybrid).
5) Show a refusal when context is insufficient.
6) Open Langfuse and show traces.

Evaluation
----------
- RAGAS: `make eval` (show `app/eval/results.json`)
- Router accuracy: `make eval-router`

Readiness
---------
- `GET /health`
- `GET /ready`

# ADR-004: Observability and Tracing
Status: Accepted

Context:
- Debugging RAG pipelines requires prompt and retrieval traces.
- The demo should show production-style observability.

Decision:
- Use Langfuse for tracing and prompt management.

Consequences:
- Traces can be inspected during demos.
- Requires Langfuse services in Docker Compose.

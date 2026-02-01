# ADR-001: Vector Database Selection
Status: Accepted

Context:
- The RAG system requires persistence and backups.
- Operations should be simple via Docker.
- The team prefers open-source components.

Decision:
- Use Qdrant as the vector database.

Alternatives:
- FAISS (no server, limited ops)
- Chroma (good dev UX, less ops maturity)
- Elasticsearch (heavyweight for POC)

Consequences:
- Qdrant provides snapshots and a stable API.
- Requires a running service container.

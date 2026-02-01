# ADR-002: Embedding Model
Status: Accepted

Context:
- The demo is English-only.
- Embeddings must run locally without paid APIs.
- Latency should be reasonable on a laptop GPU/CPU.

Decision:
- Use BAAI/bge-small-en-v1.5 for embeddings.

Alternatives:
- intfloat/e5-small-v2
- all-MiniLM-L6-v2

Consequences:
- Good English retrieval quality.
- Low resource footprint.

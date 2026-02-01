# ADR-005: Retrieval Strategy
Status: Accepted

Context:
- Retrieval should balance relevance and diversity.
- The system must support alternative modes for demos.

Decision:
- Use MMR as default retrieval strategy.
- Support hybrid retrieval as a demo option.

Consequences:
- Better coverage of distinct sources.
- Adds tunable parameters (fetch_k, lambda, alpha).

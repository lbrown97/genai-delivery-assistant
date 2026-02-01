# ADR-007: Prompt Management
Status: Accepted

Context:
- Prompts change frequently during tuning.
- We need versioned, file-based prompts.

Decision:
- Store prompts in app/llm/prompts with clear naming.

Consequences:
- Prompt changes are diffable and reviewable.

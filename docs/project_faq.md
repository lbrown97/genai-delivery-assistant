Project FAQ
===========

Why this project?
-----------------
It demonstrates end-to-end delivery skills: RAG, agent routing, guardrails, evals,
observability, and core DevOps practices with an open-source stack.

How is hallucination controlled?
--------------------------------
- Retrieval gate (min docs, unique sources, score threshold)
- Mandatory citations (refuse if citations are missing or invalid)
- Structured outputs validated with Guardrails + Pydantic

How is data protected?
----------------------
- Best-effort PII redaction on inputs and ingested content
- No customer data required; included artifacts are synthetic

What is “agentic” here?
-----------------------
The router selects tools (ADR, solution outline, user stories, risk assessment)
based on the user request and returns structured outputs with sources.

How do you evaluate quality?
----------------------------
- RAGAS metrics (faithfulness, answer relevancy, context precision/recall)
- Router accuracy dataset

What would you harden for production?
-------------------------------------
- Robust testing and CI/CD gates
- Better PII detection (NLP models)
- More precise policy guardrails
- Authn/authz and rate limiting

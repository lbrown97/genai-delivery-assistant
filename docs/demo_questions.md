Demo Questions
==============

General RAG (with citations)
----------------------------
1) What security assumptions apply to this project?
2) How are backups handled in operations?
3) Who is allowed production access?
4) What observability requirements are described?
5) What requirements are stated for the vector DB?
6) How must incidents be documented?
7) What are the availability and latency SLOs?
8) What change management rules apply?

Agentic Structured Outputs
--------------------------
1) Create a solution outline including architecture, risks, and assumptions.
2) Write an ADR for the vector database selection.
3) Generate user stories and acceptance criteria for incident handling.
4) Create a risk assessment and mitigations for this project.

ADR-focused
-----------
1) Summarize the decision on embeddings and its alternatives.
2) What is the ADR decision for guardrails and PII redaction?
3) What is the ADR decision for observability?
4) What retrieval strategy is the default and why?
5) How are PDFs ingested according to the ADRs?

Runbooks & Ops
-------------
1) What is the backup schedule and retention policy?
2) What is the incident response SLA for Sev-1 and Sev-2?
3) What steps are required for deployment?
4) What must be validated after ingestion runs?
5) What ops KPIs are tracked?

Security & Compliance
---------------------
1) What does the security logging policy require?
2) What is the access control principle for systems?
3) How long are logs and traces retained?
4) When must incidents be logged?
5) What is the severity classification scheme?

Templates & Delivery Artifacts
------------------------------
1) What does the Confluence ADR template include?
2) What is the standard Jira story format?
3) Provide a solution outline using the template structure.
4) Provide user stories using the template format.

PDF-based (external artifacts)
------------------------------
OWASP LLM Top 10 2025
1) What is LLM01 in the OWASP Top 10 for LLM Applications 2025?
2) What is LLM06 in the OWASP Top 10 for LLM Applications 2025?
3) Name two items from the OWASP Top 10 list.

NIST AI RMF 1.0
1) What are the four core functions of the AI RMF?
2) How does the AI RMF describe itself as a document?
3) Why is AI risk management important according to the AI RMF?

Google SRE Capacity Management
1) What should you determine about demand?
2) What growth types are mentioned?
3) What is a key consideration related to scaling?

Structured prompt examples (for the UI)
---------------------------------------
ADR:
```
Create an ADR.
Decision: Use Qdrant as vector database for RAG retrieval
Alternatives: FAISS, Chroma, Elasticsearch
Context query: vector database choice, operational requirements, backups, scaling
```

Solution Outline:
```
Create a solution outline.
Request: Create a solution outline including architecture, risks, and assumptions.
Context query: architecture notes, runbooks, security policy
```

User Stories:
```
Create user stories and acceptance criteria.
Request: Create user stories for incident handling and on-call access.
Context query: runbooks, incident handling, access policy
```

Risk Assessment:
```
Create a risk assessment.
Request: Create a risk assessment and mitigations for this project.
Context query: security policy, access controls, data retention
```

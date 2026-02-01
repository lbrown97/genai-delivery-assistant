# Statement of Work (Excerpt)
Scope:
- Provide a GenAI Delivery Assistant for internal consulting teams.
- Support retrieval from project artifacts (SOW, runbooks, security policies, ADRs).
- Deliver an agentic interface with structured outputs for ADRs, user stories, and risk assessments.

Non-Goals:
- No production data ingestion during POC.
- No external API calls to paid LLM providers.

Constraints:
- Must run locally via Docker Compose.
- No PII stored in logs.

Acceptance Criteria:
- Agent answers with citations or "I don't know".
- Ingestion supports PDF, Markdown, and TXT.
- Basic evaluation metrics reported.

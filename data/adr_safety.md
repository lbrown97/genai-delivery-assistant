# ADR-008: Safety Refusals
Status: Accepted

Context:
- The system must refuse unsupported or unsafe requests.
- We require grounded answers with citations.

Decision:
- If citations are missing or context is insufficient, respond with refusal.

Consequences:
- Reduced hallucinations.
- More conservative answers.

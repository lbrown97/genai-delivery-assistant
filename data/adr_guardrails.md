# ADR-003: Guardrails and PII Redaction
Status: Accepted

Context:
- Outputs must be structured and safe.
- PII should not leak into logs or traces.

Decision:
- Use GuardrailsAI for schema validation.
- Use Presidio for PII redaction with regex recognizers.

Consequences:
- Structured outputs are validated.
- PII redaction is best-effort and configurable.

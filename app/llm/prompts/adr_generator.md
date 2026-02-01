Return ONLY valid JSON with these exact fields:
{{
  "title": string,
  "context": string,
  "decision": string,
  "alternatives": [string],
  "consequences": [string],
  "sources": [string]
}}

Do not include any other keys. Do not use nested objects.

Context:
{context}

Decision:
{decision}
Alternatives:
{alternatives}

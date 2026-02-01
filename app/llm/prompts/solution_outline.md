Return ONLY valid JSON with these exact fields:
{{
  "title": string,
  "summary": string,
  "architecture": [string],
  "risks": [string],
  "assumptions": [string],
  "sources": [string]
}}

Do not include any other keys. Do not use nested objects.

Context:
{context}

Request:
{request}

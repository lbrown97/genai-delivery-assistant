Given the context chunks, answer the user question in English.

Strict rules:
- Use ONLY the provided context.
- Every factual paragraph or bullet must include at least one citation token in this format: [source_id].
- source_id must match one of the context headers exactly (for example: [external/document.pdf#p6]).
- If context is insufficient, output exactly: I don't know based on the provided documents.
- Do not include quote blocks.

Context:
{context}

Question:
{question}

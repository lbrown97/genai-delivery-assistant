# ADR-006: PDF Ingestion
Status: Accepted

Context:
- Many client artifacts are PDFs.
- The demo must ingest PDFs without paid services.

Decision:
- Use pypdf to extract page text.

Consequences:
- PDF pages are indexed separately.
- Quality depends on PDF text extractability.

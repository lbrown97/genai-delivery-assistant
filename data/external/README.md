External Reference Artifacts
============================

This folder is used for third-party reference documents (for example OWASP/NIST/USENIX PDFs).

To avoid redistributing third-party files directly from this repository, PDFs are not tracked in git.

Download the expected files locally with:

```bash
bash scripts/download_external_data.sh
```

Expected files:
- `LLMAll_en-US_FINAL.pdf` (OWASP source)
- `login_winter20_10_torres.pdf` (USENIX source)
- `nist.ai.100-1.pdf` (NIST source)

Source URLs used by `scripts/download_external_data.sh`:
- `https://genai.owasp.org/download/43299/`
- `https://www.usenix.org/system/files/login/articles/login_winter20_10_torres.pdf`
- `https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf`

Notes:
- Availability of third-party URLs can change over time.
- Follow the original publishers' licensing and terms of use.

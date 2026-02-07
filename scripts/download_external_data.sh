#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/data/external"

mkdir -p "$OUT_DIR"

download() {
  local url="$1"
  local filename="$2"
  echo "Downloading $filename"
  curl -fL --retry 3 --retry-delay 2 "$url" -o "$OUT_DIR/$filename"
}

download "https://genai.owasp.org/download/43299/" "LLMAll_en-US_FINAL.pdf"
download "https://www.usenix.org/system/files/login/articles/login_winter20_10_torres.pdf" "login_winter20_10_torres.pdf"
download "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf" "nist.ai.100-1.pdf"

echo "External reference PDFs saved to: $OUT_DIR"

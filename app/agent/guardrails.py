import logging
import os
from typing import Any, Type

_PRESIDIO_LOGGERS = ["presidio-analyzer", "presidio_analyzer"]
for _name in _PRESIDIO_LOGGERS:
    logging.getLogger(_name).setLevel(logging.WARNING)


def groundedness_with_scores(
    docs,
    scores,
    *,
    min_docs: int = 1,
    min_unique_sources: int = 1,
    min_score: float = 0.25,
) -> bool:
    if docs is None or scores is None:
        return False
    if len(docs) < min_docs:
        return False
    unique_sources = {d.metadata.get("source_id", "") for d in docs}
    if len([s for s in unique_sources if s]) < min_unique_sources:
        return False
    if not scores:
        return False
    # Lower distance is better for cosine distance; accept if any score <= threshold
    return min(scores) <= min_score


def _pii_enabled() -> bool:
    return os.getenv("PII_REDACTION", "1").lower() in {"1", "true", "yes", "on"}


def redact_pii(text: str) -> str:
    """
    Best-effort PII redaction using Presidio if available.
    Falls back to raw text if Presidio isn't installed or misconfigured.
    """
    if not _pii_enabled():
        return text

    try:
        from presidio_analyzer import (
            AnalyzerEngine,
            Pattern,
            PatternRecognizer,
            RecognizerRegistry,
        )
        from presidio_anonymizer import AnonymizerEngine
    except Exception:
        return text

    analyzer, anonymizer = _get_presidio_engines(
        AnalyzerEngine,
        AnonymizerEngine,
        RecognizerRegistry,
        PatternRecognizer,
        Pattern,
    )
    results = analyzer.analyze(text=text, language="en")
    if not results:
        return text
    return anonymizer.anonymize(text=text, analyzer_results=results).text


def redact_pii_any(value: Any) -> Any:
    """
    Recursively redact PII in strings within dict/list structures.
    """
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, list):
        return [redact_pii_any(v) for v in value]
    if isinstance(value, dict):
        return {k: redact_pii_any(v) for k, v in value.items()}
    return value


_PRESIDIO_CACHE: dict[str, Any] = {"analyzer": None, "anonymizer": None}


def _get_presidio_engines(
    AnalyzerEngine,
    AnonymizerEngine,
    RecognizerRegistry,
    PatternRecognizer,
    Pattern,
):
    if _PRESIDIO_CACHE["analyzer"] and _PRESIDIO_CACHE["anonymizer"]:
        return _PRESIDIO_CACHE["analyzer"], _PRESIDIO_CACHE["anonymizer"]

    registry = RecognizerRegistry()
    email_pattern = Pattern(
        name="email",
        regex=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        score=0.85,
    )
    phone_pattern = Pattern(
        name="phone",
        regex=r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}",
        score=0.6,
    )
    ip_pattern = Pattern(
        name="ip_address",
        regex=r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        score=0.6,
    )
    api_key_pattern = Pattern(
        name="api_key",
        regex=r"\b(?:sk|pk|api|key|token)[-_]?[A-Za-z0-9]{12,}\b",
        score=0.6,
    )
    iban_pattern = Pattern(
        name="iban",
        regex=r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b",
        score=0.6,
    )

    registry.add_recognizer(
        PatternRecognizer(supported_entity="EMAIL_ADDRESS", patterns=[email_pattern])
    )
    registry.add_recognizer(
        PatternRecognizer(supported_entity="PHONE_NUMBER", patterns=[phone_pattern])
    )
    registry.add_recognizer(
        PatternRecognizer(supported_entity="IP_ADDRESS", patterns=[ip_pattern])
    )
    registry.add_recognizer(
        PatternRecognizer(supported_entity="API_KEY", patterns=[api_key_pattern])
    )
    registry.add_recognizer(
        PatternRecognizer(supported_entity="IBAN", patterns=[iban_pattern])
    )

    analyzer = AnalyzerEngine(registry=registry, supported_languages=["en"])
    anonymizer = AnonymizerEngine()

    _PRESIDIO_CACHE["analyzer"] = analyzer
    _PRESIDIO_CACHE["anonymizer"] = anonymizer
    return analyzer, anonymizer


def _guard_from_pydantic(model: Type[Any]):
    try:
        from guardrails import Guard
    except Exception:
        return None
    try:
        return Guard.from_pydantic(model)
    except Exception:
        return None


def parse_with_guardrails(model: Type[Any], raw: str):
    """
    Validate LLM JSON with Guardrails if available; fall back to Pydantic.
    """
    guard = _guard_from_pydantic(model)
    if guard is None:
        return model.model_validate_json(raw)
    try:
        parsed = guard.parse(raw)
        # guard.parse may return raw dict or a GuardrailsOutput; normalize
        if hasattr(parsed, "validated_output"):
            return model.model_validate(parsed.validated_output)
        if isinstance(parsed, dict):
            return model.model_validate(parsed)
        return model.model_validate_json(raw)
    except Exception:
        return model.model_validate_json(raw)


def validate_citations(text: str, allowed_source_ids: set[str] | None = None) -> bool:
    """
    Minimal citation format check: require at least one [source_id]-style token.
    """
    import re
    matches = re.findall(r"\[([^\[\]\n]{1,100})\]", text)
    if not matches:
        return False
    if allowed_source_ids is None:
        return True
    return any(m in allowed_source_ids for m in matches)

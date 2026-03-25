from app.agent.guardrails import (
    groundedness_with_scores,
    parse_with_guardrails,
    redact_pii,
    redact_pii_any,
    validate_citations,
)
from app.agent.router_prompts import ANSWER, SYSTEM
from app.agent.tools import retrieve_context_with_scores
from app.core.env import env_float
from app.core.observability import get_langfuse_handler
from app.llm.models import get_chat_model
from app.rag.retriever import get_active_doc_scope

UNKNOWN_ANSWER = "I don't know based on the provided documents."


def groundedness_min_score(default: float) -> float:
    """Resolve min-score threshold by active doc scope for stable demo behavior."""

    scope = get_active_doc_scope()
    if scope == "project":
        return env_float("GROUNDEDNESS_MIN_SCORE_PROJECT", 0.65)
    if scope == "external":
        return env_float("GROUNDEDNESS_MIN_SCORE_EXTERNAL", 0.8)
    return env_float("GROUNDEDNESS_MIN_SCORE_ALL", default)


def lf_config(tags: list[str], metadata: dict, extra_meta: dict | None = None):
    """Build LangChain invocation config with optional Langfuse callback."""

    handler = get_langfuse_handler()
    if not handler:
        return {}
    if extra_meta:
        metadata = {**metadata, **extra_meta}
    return {
        "callbacks": [handler],
        "tags": tags,
        "metadata": metadata,
    }


def wrap_response(
    *,
    agent_tool: str,
    agent_args: dict,
    sources: list | None = None,
    answer: str | None = None,
    structured: dict | None = None,
    error: str | None = None,
    message: str | None = None,
):
    """Create a normalized API response payload with redaction safeguards."""

    payload = {
        "agent_tool": agent_tool,
        "agent_args": agent_args,
        "sources": sources or [],
    }
    if answer is not None:
        payload["answer"] = redact_pii(answer)
    if structured is not None:
        payload["structured"] = redact_pii_any(structured)
    if error:
        payload["error"] = error
    if message:
        payload["message"] = message
    return payload


def retrieve_context(
    context_query: str,
    *,
    gate: dict,
    agent_tool: str,
    agent_args: dict,
    error_message: str,
):
    """Retrieve context and enforce groundedness before tool execution."""

    r = retrieve_context_with_scores(context_query)
    docs = r["docs"]
    min_score = groundedness_min_score(gate["min_score"])
    if not groundedness_with_scores(
        docs,
        r.get("scores"),
        min_docs=gate["min_docs"],
        min_unique_sources=gate.get("min_unique_sources", 1),
        min_score=min_score,
    ):
        return None, wrap_response(
            agent_tool=agent_tool,
            agent_args=agent_args,
            sources=[],
            error="not_enough_context",
            message=error_message,
        )
    return (r, docs), None


def finalize_sources(model_obj, docs):
    """Restrict structured-output sources to IDs present in retrieved docs."""

    known_ids = {d.metadata.get("source_id", "") for d in docs}
    model_obj.sources = [s for s in model_obj.sources if s in known_ids] or list(known_ids)[:3]
    return model_obj


def generate_structured(
    *,
    agent_tool: str,
    prompt_template: str,
    model_cls,
    format_kwargs: dict,
    agent_args: dict,
    context_query: str,
    gate: dict,
    temperature: float,
    tags: list[str],
    agent_meta: dict | None,
    error_message: str,
):
    """Run the shared structured-generation workflow for a tool."""

    retrieved, fail = retrieve_context(
        context_query,
        gate=gate,
        agent_tool=agent_tool,
        agent_args=agent_args,
        error_message=error_message,
    )
    if fail:
        return fail

    r, docs = retrieved
    llm = get_chat_model(temperature=temperature, format="json")
    prompt = f"{SYSTEM}\n\n" + prompt_template.format(context=r["context"], **format_kwargs)
    cfg = lf_config(
        tags=tags,
        metadata={"customer": "demo", "retrieved_docs": len(docs)},
        extra_meta=agent_meta,
    )
    msg = llm.invoke(prompt, config=cfg)
    raw = (msg.content or "").strip()
    if not raw:
        return wrap_response(
            agent_tool=agent_tool,
            agent_args=agent_args,
            sources=r["sources"],
            error="empty_model_output",
            message="Model returned an empty structured response.",
        )
    try:
        obj = parse_with_guardrails(model_cls, raw)
    except Exception:
        return wrap_response(
            agent_tool=agent_tool,
            agent_args=agent_args,
            sources=r["sources"],
            error="invalid_structured_output",
            message="Model output was not valid structured JSON.",
        )
    obj = finalize_sources(obj, docs)
    return wrap_response(
        agent_tool=agent_tool,
        agent_args=agent_args,
        sources=r["sources"],
        structured=obj.model_dump(),
    )


def is_unknown_answer(text: str) -> bool:
    """Check whether model output starts with the canonical unknown answer."""

    normalized = " ".join((text or "").strip().split()).lower()
    target = UNKNOWN_ANSWER.rstrip(".").lower()
    return normalized.startswith(target)


def build_answer_prompt(context: str, safe_question: str, source_ids: list[str]) -> str:
    """Build ask-mode prompt with explicit source allowlist for citations."""

    allowed_sources_block = "\n".join(f"- [{sid}]" for sid in source_ids)
    return (
        f"{SYSTEM}\n\n"
        + ANSWER.format(context=context, question=safe_question)
        + "\n\nAllowed source_ids for citations (use only these):\n"
        + (allowed_sources_block or "- (none)")
    )


def generate_ask_draft(
    question: str,
    *,
    context: str,
    sources: list[dict],
    temperature: float,
    config: dict | None = None,
) -> str:
    """Generate a raw ask-mode draft answer using the shared app prompt."""

    safe_question = redact_pii(question)
    source_ids = [s.get("source_id", "") for s in sources if s.get("source_id")]
    prompt = build_answer_prompt(context, safe_question, source_ids)
    llm = get_chat_model(temperature=temperature)
    if config:
        msg = llm.invoke(prompt, config=config)
    else:
        msg = llm.invoke(prompt)
    return msg.content


def normalize_ask_answer(answer_text: str, sources: list[dict]) -> tuple[str, str | None]:
    """Normalize ask-mode answer and validate citations against retrieved sources."""

    if is_unknown_answer(answer_text):
        return UNKNOWN_ANSWER, None

    allowed_ids = {s.get("source_id", "") for s in sources}
    if not validate_citations(answer_text, allowed_source_ids=allowed_ids):
        return UNKNOWN_ANSWER, "citation_validation_failed"

    return answer_text, None

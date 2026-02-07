from app.agent.guardrails import (
    groundedness_with_scores,
    parse_with_guardrails,
    redact_pii,
    redact_pii_any,
)
from app.agent.router_prompts import SYSTEM
from app.agent.tools import retrieve_context_with_scores
from app.core.observability import get_langfuse_handler
from app.llm.models import get_chat_model


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
    if not groundedness_with_scores(
        docs,
        r.get("scores"),
        min_docs=gate["min_docs"],
        min_unique_sources=gate.get("min_unique_sources", 1),
        min_score=gate["min_score"],
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
    llm = get_chat_model(temperature=temperature)
    prompt = f"{SYSTEM}\n\n" + prompt_template.format(context=r["context"], **format_kwargs)
    cfg = lf_config(
        tags=tags,
        metadata={"customer": "demo", "retrieved_docs": len(docs)},
        extra_meta=agent_meta,
    )
    msg = llm.invoke(prompt, config=cfg)
    obj = parse_with_guardrails(model_cls, msg.content)
    obj = finalize_sources(obj, docs)
    return wrap_response(
        agent_tool=agent_tool,
        agent_args=agent_args,
        sources=r["sources"],
        structured=obj.model_dump(),
    )

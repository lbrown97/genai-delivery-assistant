import json
from pathlib import Path

from app.agent.guardrails import (
    groundedness_with_scores,
    parse_with_guardrails,
    redact_pii,
    redact_pii_any,
    validate_citations,
)
from app.agent.schemas import ADR, RiskAssessment, SolutionOutline, ToolCall, UserStories
from app.agent.tools import retrieve_context_with_scores
from app.core.observability import get_langfuse_handler
from app.llm.models import get_chat_model

PROMPTS_DIR = Path("app/llm/prompts")


def _read_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


SYSTEM = _read_prompt("system.md")
ANSWER = _read_prompt("answer_with_citations.md")
ADR_PROMPT = _read_prompt("adr_generator.md")
SOLUTION_PROMPT = _read_prompt("solution_outline.md")
STORIES_PROMPT = _read_prompt("user_stories.md")
RISK_PROMPT = _read_prompt("risk_assessment.md")
ROUTER_PROMPT = _read_prompt("tool_router.md")


def _lf_config(tags: list[str], metadata: dict, extra_meta: dict | None = None):
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


def _wrap_response(
    *,
    agent_tool: str,
    agent_args: dict,
    sources: list | None = None,
    answer: str | None = None,
    structured: dict | None = None,
    error: str | None = None,
    message: str | None = None,
):
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


def answer_question(question: str, agent_meta: dict | None = None):
    safe_question = redact_pii(question)
    r = retrieve_context_with_scores(question)
    docs = r["docs"]

    if not groundedness_with_scores(
        docs,
        r.get("scores"),
        min_docs=1,
        min_unique_sources=1,
        min_score=0.25,
    ):
        return _wrap_response(
            agent_tool="ask",
            agent_args={"question": safe_question},
            sources=[],
            answer="I don't know based on the provided documents.",
        )

    llm = get_chat_model(temperature=0.2)
    prompt = f"{SYSTEM}\n\n" + ANSWER.format(context=r["context"], question=safe_question)

    cfg = _lf_config(
        tags=["use_case:ask"],
        metadata={"customer": "demo", "retrieved_docs": len(docs)},
        extra_meta=agent_meta,
    )
    msg = llm.invoke(prompt, config=cfg)

    allowed_ids = {s.get("source_id", "") for s in r["sources"]}
    if not validate_citations(msg.content, allowed_source_ids=allowed_ids):
        return _wrap_response(
            agent_tool="ask",
            agent_args={"question": safe_question},
            sources=[],
            answer="I don't know based on the provided documents.",
        )

    return _wrap_response(
        agent_tool="ask",
        agent_args={"question": safe_question},
        sources=r["sources"],
        answer=msg.content,
    )


def generate_adr(
    decision: str,
    alternatives: list[str],
    question_for_context: str,
    agent_meta: dict | None = None,
):
    safe_decision = redact_pii(decision)
    safe_alternatives = [redact_pii(a) for a in alternatives]
    safe_context_query = redact_pii(question_for_context)
    r = retrieve_context_with_scores(question_for_context)
    docs = r["docs"]

    if not groundedness_with_scores(
        docs,
        r.get("scores"),
        min_docs=2,
        min_unique_sources=1,
        min_score=0.25,
    ):
        return _wrap_response(
            agent_tool="adr",
            agent_args={
                "decision": safe_decision,
                "alternatives": safe_alternatives,
                "context_query": safe_context_query,
            },
            sources=[],
            error="not_enough_context",
            message="Not enough context from documents for ADR.",
        )

    llm = get_chat_model(temperature=0.1)
    prompt = f"{SYSTEM}\n\n" + ADR_PROMPT.format(
        context=r["context"],
        decision=safe_decision,
        alternatives=json.dumps(safe_alternatives, ensure_ascii=False),
    )

    cfg = _lf_config(
        tags=["use_case:adr"],
        metadata={"customer": "demo", "retrieved_docs": len(docs)},
        extra_meta=agent_meta,
    )
    msg = llm.invoke(prompt, config=cfg)

    adr = parse_with_guardrails(ADR, msg.content)

    known_ids = {d.metadata.get("source_id", "") for d in docs}
    adr.sources = [s for s in adr.sources if s in known_ids] or list(known_ids)[:3]

    return _wrap_response(
        agent_tool="adr",
        agent_args={
            "decision": safe_decision,
            "alternatives": safe_alternatives,
            "context_query": safe_context_query,
        },
        sources=r["sources"],
        structured=adr.model_dump(),
    )


def generate_solution_outline(request: str, context_query: str, agent_meta: dict | None = None):
    safe_request = redact_pii(request)
    safe_context_query = redact_pii(context_query)
    r = retrieve_context_with_scores(context_query)
    docs = r["docs"]

    if not groundedness_with_scores(
        docs,
        r.get("scores"),
        min_docs=2,
        min_unique_sources=1,
        min_score=0.25,
    ):
        return _wrap_response(
            agent_tool="solution_outline",
            agent_args={"request": safe_request, "context_query": safe_context_query},
            sources=[],
            error="not_enough_context",
            message="Not enough context from documents for Solution Outline.",
        )

    llm = get_chat_model(temperature=0.2)
    prompt = f"{SYSTEM}\n\n" + SOLUTION_PROMPT.format(
        context=r["context"],
        request=safe_request,
    )

    cfg = _lf_config(
        tags=["use_case:solution_outline"],
        metadata={"customer": "demo", "retrieved_docs": len(docs)},
        extra_meta=agent_meta,
    )
    msg = llm.invoke(prompt, config=cfg)

    outline = parse_with_guardrails(SolutionOutline, msg.content)

    known_ids = {d.metadata.get("source_id", "") for d in docs}
    outline.sources = [s for s in outline.sources if s in known_ids] or list(known_ids)[:3]

    return _wrap_response(
        agent_tool="solution_outline",
        agent_args={"request": safe_request, "context_query": safe_context_query},
        sources=r["sources"],
        structured=outline.model_dump(),
    )


def generate_user_stories(request: str, context_query: str, agent_meta: dict | None = None):
    safe_request = redact_pii(request)
    safe_context_query = redact_pii(context_query)
    r = retrieve_context_with_scores(context_query)
    docs = r["docs"]

    if not groundedness_with_scores(
        docs,
        r.get("scores"),
        min_docs=1,
        min_unique_sources=1,
        min_score=0.25,
    ):
        return _wrap_response(
            agent_tool="user_stories",
            agent_args={"request": safe_request, "context_query": safe_context_query},
            sources=[],
            error="not_enough_context",
            message="Not enough context from documents for User Stories.",
        )

    llm = get_chat_model(temperature=0.2)
    prompt = f"{SYSTEM}\n\n" + STORIES_PROMPT.format(
        context=r["context"],
        request=safe_request,
    )

    cfg = _lf_config(
        tags=["use_case:user_stories"],
        metadata={"customer": "demo", "retrieved_docs": len(docs)},
        extra_meta=agent_meta,
    )
    msg = llm.invoke(prompt, config=cfg)

    stories = parse_with_guardrails(UserStories, msg.content)

    known_ids = {d.metadata.get("source_id", "") for d in docs}
    stories.sources = [s for s in stories.sources if s in known_ids] or list(known_ids)[:3]

    return _wrap_response(
        agent_tool="user_stories",
        agent_args={"request": safe_request, "context_query": safe_context_query},
        sources=r["sources"],
        structured=stories.model_dump(),
    )


def generate_risk_assessment(request: str, context_query: str, agent_meta: dict | None = None):
    safe_request = redact_pii(request)
    safe_context_query = redact_pii(context_query)
    r = retrieve_context_with_scores(context_query)
    docs = r["docs"]

    if not groundedness_with_scores(
        docs,
        r.get("scores"),
        min_docs=1,
        min_unique_sources=1,
        min_score=0.25,
    ):
        return _wrap_response(
            agent_tool="risk_assessment",
            agent_args={"request": safe_request, "context_query": safe_context_query},
            sources=[],
            error="not_enough_context",
            message="Not enough context from documents for Risk Assessment.",
        )

    llm = get_chat_model(temperature=0.2)
    prompt = f"{SYSTEM}\n\n" + RISK_PROMPT.format(
        context=r["context"],
        request=safe_request,
    )

    cfg = _lf_config(
        tags=["use_case:risk_assessment"],
        metadata={"customer": "demo", "retrieved_docs": len(docs)},
        extra_meta=agent_meta,
    )
    msg = llm.invoke(prompt, config=cfg)

    risk = parse_with_guardrails(RiskAssessment, msg.content)

    known_ids = {d.metadata.get("source_id", "") for d in docs}
    risk.sources = [s for s in risk.sources if s in known_ids] or list(known_ids)[:3]

    return _wrap_response(
        agent_tool="risk_assessment",
        agent_args={"request": safe_request, "context_query": safe_context_query},
        sources=r["sources"],
        structured=risk.model_dump(),
    )


def agent_route(question: str):
    # Route to best tool using a deterministic call
    safe_question = redact_pii(question)
    llm = get_chat_model(temperature=0.0)
    prompt = f"{SYSTEM}\n\n" + ROUTER_PROMPT.format(question=safe_question)
    cfg = _lf_config(
        tags=["use_case:router"],
        metadata={"customer": "demo"},
    )
    msg = llm.invoke(prompt, config=cfg)

    try:
        call = ToolCall.model_validate_json(msg.content)
    except Exception:
        # Fallback to plain RAG answer if routing fails
        return answer_question(question)

    tool = call.tool
    args = call.args or {}

    # Record router decision as an explicit Langfuse event (if available)
    handler = get_langfuse_handler()
    if handler:
        try:
            handler.on_event(
                "router_decision",
                metadata={"router_tool": tool, "router_args": args},
            )
        except Exception:
            pass

    agent_meta = {"router_tool": tool, "router_args": args}

    if tool == "ask":
        out = answer_question(args.get("question", question), agent_meta=agent_meta)
    elif tool == "adr":
        out = generate_adr(
            args.get("decision", question),
            args.get("alternatives", ["Option A", "Option B", "Option C"]),
            args.get("context_query", question),
            agent_meta=agent_meta,
        )
    elif tool == "solution_outline":
        out = generate_solution_outline(
            args.get("request", question),
            args.get("context_query", question),
            agent_meta=agent_meta,
        )
    elif tool == "user_stories":
        out = generate_user_stories(
            args.get("request", question),
            args.get("context_query", question),
            agent_meta=agent_meta,
        )
    elif tool == "risk_assessment":
        out = generate_risk_assessment(
            args.get("request", question),
            args.get("context_query", question),
            agent_meta=agent_meta,
        )
    else:
        out = answer_question(question, agent_meta=agent_meta)
        tool = "ask"

    return out

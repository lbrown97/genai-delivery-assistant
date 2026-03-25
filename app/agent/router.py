import json

from app.agent.guardrails import groundedness_with_scores, redact_pii
from app.agent.router_prompts import ROUTER_PROMPT
from app.agent.router_utils import (
    UNKNOWN_ANSWER,
    generate_ask_draft,
    generate_structured,
    groundedness_min_score,
    lf_config,
    normalize_ask_answer,
    wrap_response,
)
from app.agent.schemas import ToolCall
from app.agent.tool_config import TOOL_CONFIG
from app.core.env import env_float
from app.llm.models import get_chat_model

ARTIFACT_TERMS = (
    "adr",
    "architecture decision",
    "decision record",
    "solution outline",
    "user story",
    "user stories",
    "acceptance criteria",
    "risk assessment",
    "risk analysis",
    "risk register",
)
CREATE_VERBS = ("create", "draft", "write", "generate", "produce", "prepare")
ARTIFACT_NOUNS = ("outline", "stories", "assessment", "adr", "decision record")


def _run_structured(
    tool_key: str,
    *,
    format_kwargs: dict,
    agent_args: dict,
    context_query: str,
    agent_meta: dict | None,
):
    """Execute a structured tool using shared configuration from `TOOL_CONFIG`."""

    cfg = TOOL_CONFIG[tool_key]
    return generate_structured(
        agent_tool=tool_key,
        prompt_template=cfg["prompt"],
        model_cls=cfg["model_cls"],
        format_kwargs=format_kwargs,
        agent_args=agent_args,
        context_query=context_query,
        gate=cfg["gate"],
        temperature=cfg["temperature"],
        tags=cfg["tags"],
        agent_meta=agent_meta,
        error_message=cfg["error_message"],
    )


def _is_artifact_request(question: str) -> bool:
    """Heuristically detect whether a question requests an artifact output."""

    q = question.lower()
    if any(term in q for term in ARTIFACT_TERMS):
        return True
    return any(v in q for v in CREATE_VERBS) and any(n in q for n in ARTIFACT_NOUNS)


def answer_question(question: str, agent_meta: dict | None = None):
    """Answer a general question with retrieval, grounding, and citation checks."""

    safe_question = redact_pii(question)
    from app.agent.tools import retrieve_context_with_scores

    r = retrieve_context_with_scores(question)
    docs = r["docs"]

    if not groundedness_with_scores(
        docs,
        r.get("scores"),
        min_docs=1,
        min_unique_sources=1,
        min_score=groundedness_min_score(0.8),
    ):
        return wrap_response(
            agent_tool="ask",
            agent_args={"question": safe_question},
            sources=[],
            error="not_enough_context",
            message="Retrieved context did not meet groundedness thresholds.",
            answer=UNKNOWN_ANSWER,
        )

    cfg = lf_config(
        tags=["use_case:ask"],
        metadata={"customer": "demo", "retrieved_docs": len(docs)},
        extra_meta=agent_meta,
    )
    draft = generate_ask_draft(
        question,
        context=r["context"],
        sources=r["sources"],
        temperature=env_float("ASK_TEMPERATURE", 0.1),
        config=cfg,
    )
    answer_text, error = normalize_ask_answer(draft, r["sources"])

    if error:
        return wrap_response(
            agent_tool="ask",
            agent_args={"question": safe_question},
            sources=r["sources"],
            error=error,
            message="Model output did not cite retrieved sources in an acceptable format.",
            answer=UNKNOWN_ANSWER,
        )

    return wrap_response(
        agent_tool="ask",
        agent_args={"question": safe_question},
        sources=r["sources"],
        answer=answer_text,
    )


def generate_adr(
    decision: str,
    alternatives: list[str],
    question_for_context: str,
    agent_meta: dict | None = None,
):
    """Generate an ADR artifact from retrieved context."""

    safe_decision = redact_pii(decision)
    safe_alternatives = [redact_pii(a) for a in alternatives]
    safe_context_query = redact_pii(question_for_context)
    return _run_structured(
        "adr",
        format_kwargs={
            "decision": safe_decision,
            "alternatives": json.dumps(safe_alternatives, ensure_ascii=False),
        },
        agent_args={
            "decision": safe_decision,
            "alternatives": safe_alternatives,
            "context_query": safe_context_query,
        },
        context_query=question_for_context,
        agent_meta=agent_meta,
    )


def generate_solution_outline(request: str, context_query: str, agent_meta: dict | None = None):
    """Generate a solution-outline artifact from retrieved context."""

    return _generate_request_artifact(
        "solution_outline",
        request=request,
        context_query=context_query,
        agent_meta=agent_meta,
    )


def generate_user_stories(request: str, context_query: str, agent_meta: dict | None = None):
    """Generate user stories and acceptance criteria from retrieved context."""

    return _generate_request_artifact(
        "user_stories",
        request=request,
        context_query=context_query,
        agent_meta=agent_meta,
    )


def generate_risk_assessment(request: str, context_query: str, agent_meta: dict | None = None):
    """Generate a risk assessment artifact from retrieved context."""

    return _generate_request_artifact(
        "risk_assessment",
        request=request,
        context_query=context_query,
        agent_meta=agent_meta,
    )


def _generate_request_artifact(
    tool_key: str,
    *,
    request: str,
    context_query: str,
    agent_meta: dict | None = None,
):
    """Shared helper for request-driven structured artifacts."""

    safe_request = redact_pii(request)
    safe_context_query = redact_pii(context_query)
    return _run_structured(
        tool_key,
        format_kwargs={"request": safe_request},
        agent_args={"request": safe_request, "context_query": safe_context_query},
        context_query=context_query,
        agent_meta=agent_meta,
    )


def select_tool_call(question: str) -> ToolCall:
    """Select a tool call using the same logic as the production router path."""

    safe_question = redact_pii(question)
    if not _is_artifact_request(safe_question):
        return ToolCall(tool="ask", args={"question": safe_question})

    llm = get_chat_model(temperature=0.0, format="json")
    prompt = ROUTER_PROMPT.format(question=safe_question)
    cfg = lf_config(
        tags=["use_case:router"],
        metadata={"customer": "demo"},
    )
    msg = llm.invoke(prompt, config=cfg)

    try:
        return ToolCall.model_validate_json(msg.content)
    except Exception:
        return ToolCall(tool="ask", args={"question": safe_question})


def agent_route(question: str):
    """Route the request to `ask` or a structured artifact tool."""

    call = select_tool_call(question)
    tool = call.tool
    args = call.args or {}

    agent_meta = {"router_tool": tool, "router_args": args}
    if tool == "ask":
        return answer_question(question, agent_meta=agent_meta)

    handlers = {
        "adr": lambda: generate_adr(
            args.get("decision", question),
            args.get("alternatives", ["Option A", "Option B", "Option C"]),
            args.get("context_query", question),
            agent_meta=agent_meta,
        ),
        "solution_outline": lambda: generate_solution_outline(
            args.get("request", question),
            args.get("context_query", question),
            agent_meta=agent_meta,
        ),
        "user_stories": lambda: generate_user_stories(
            args.get("request", question),
            args.get("context_query", question),
            agent_meta=agent_meta,
        ),
        "risk_assessment": lambda: generate_risk_assessment(
            args.get("request", question),
            args.get("context_query", question),
            agent_meta=agent_meta,
        ),
    }

    handler = handlers.get(tool)
    if handler is None:
        return answer_question(question, agent_meta=agent_meta)
    return handler()

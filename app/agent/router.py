import json

from app.agent.guardrails import groundedness_with_scores, redact_pii, validate_citations
from app.agent.router_prompts import ANSWER, ROUTER_PROMPT, SYSTEM
from app.agent.router_utils import generate_structured, lf_config, wrap_response
from app.agent.schemas import ToolCall
from app.llm.models import get_chat_model


def _run_structured(
    tool_key: str,
    *,
    format_kwargs: dict,
    agent_args: dict,
    context_query: str,
    agent_meta: dict | None,
):
    from app.agent.tool_config import TOOL_CONFIG

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
    q = question.lower()
    artifact_terms = [
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
    ]
    if any(term in q for term in artifact_terms):
        return True
    create_verbs = ["create", "draft", "write", "generate", "produce", "prepare"]
    artifact_nouns = ["outline", "stories", "assessment", "adr", "decision record"]
    return any(v in q for v in create_verbs) and any(n in q for n in artifact_nouns)


def answer_question(question: str, agent_meta: dict | None = None):
    safe_question = redact_pii(question)
    from app.agent.tools import retrieve_context_with_scores

    r = retrieve_context_with_scores(question)
    docs = r["docs"]

    if not groundedness_with_scores(
        docs,
        r.get("scores"),
        min_docs=1,
        min_unique_sources=1,
        min_score=0.8,
    ):
        return wrap_response(
            agent_tool="ask",
            agent_args={"question": safe_question},
            sources=[],
            answer="I don't know based on the provided documents.",
        )

    llm = get_chat_model(temperature=0.2)
    prompt = f"{SYSTEM}\n\n" + ANSWER.format(context=r["context"], question=safe_question)

    cfg = lf_config(
        tags=["use_case:ask"],
        metadata={"customer": "demo", "retrieved_docs": len(docs)},
        extra_meta=agent_meta,
    )
    msg = llm.invoke(prompt, config=cfg)

    allowed_ids = {s.get("source_id", "") for s in r["sources"]}
    if not validate_citations(msg.content, allowed_source_ids=allowed_ids):
        return wrap_response(
            agent_tool="ask",
            agent_args={"question": safe_question},
            sources=[],
            answer="I don't know based on the provided documents.",
        )

    return wrap_response(
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
    safe_request = redact_pii(request)
    safe_context_query = redact_pii(context_query)
    return _run_structured(
        "solution_outline",
        format_kwargs={"request": safe_request},
        agent_args={"request": safe_request, "context_query": safe_context_query},
        context_query=context_query,
        agent_meta=agent_meta,
    )


def generate_user_stories(request: str, context_query: str, agent_meta: dict | None = None):
    safe_request = redact_pii(request)
    safe_context_query = redact_pii(context_query)
    return _run_structured(
        "user_stories",
        format_kwargs={"request": safe_request},
        agent_args={"request": safe_request, "context_query": safe_context_query},
        context_query=context_query,
        agent_meta=agent_meta,
    )


def generate_risk_assessment(request: str, context_query: str, agent_meta: dict | None = None):
    safe_request = redact_pii(request)
    safe_context_query = redact_pii(context_query)
    return _run_structured(
        "risk_assessment",
        format_kwargs={"request": safe_request},
        agent_args={"request": safe_request, "context_query": safe_context_query},
        context_query=context_query,
        agent_meta=agent_meta,
    )


def agent_route(question: str):
    # Route to best tool using a deterministic call
    safe_question = redact_pii(question)
    if not _is_artifact_request(safe_question):
        return answer_question(question)

    llm = get_chat_model(temperature=0.0)
    prompt = ROUTER_PROMPT.format(question=safe_question)
    cfg = lf_config(
        tags=["use_case:router"],
        metadata={"customer": "demo"},
    )
    msg = llm.invoke(prompt, config=cfg)

    call = ToolCall.model_validate_json(msg.content)

    tool = call.tool
    args = call.args or {}

    agent_meta = {"router_tool": tool, "router_args": args}

    if tool == "adr":
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

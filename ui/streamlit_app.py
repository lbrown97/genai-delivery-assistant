import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")
RETRIEVAL_OPTIONS = ["mmr", "hybrid", "similarity"]
SCOPE_OPTIONS = ["project", "all", "external"]

st.set_page_config(page_title="GenAI Delivery Assistant", layout="wide")
st.title("GenAI Delivery Assistant (Agentic)")

col1, col2 = st.columns([1, 2])


def _render_response(payload: dict):
    """Render agent response sections in a consistent Streamlit layout."""

    if not isinstance(payload, dict):
        st.json(payload)
        return

    answer = payload.get("answer")
    structured = payload.get("structured")
    if answer:
        st.markdown(answer)
    elif structured:
        st.json(structured)

    if payload.get("message"):
        st.info(payload["message"])
    if payload.get("error"):
        st.error(payload["error"])

    with st.expander("Sources", expanded=False):
        st.json(payload.get("sources", []))

    with st.expander("Agent Metadata", expanded=False):
        st.code(
            {
                "agent_tool": payload.get("agent_tool"),
                "agent_args": payload.get("agent_args"),
            },
            language="json",
        )

    with st.expander("Raw JSON", expanded=False):
        st.json(payload)


with col1:
    st.subheader("Ingest")
    if st.button("Ingest /data"):
        r = requests.post(f"{API_URL}/ingest", timeout=300)
        st.json(r.json())

with col2:
    st.subheader("Ask Agent")
    mode = st.selectbox(
        "Retrieval mode",
        RETRIEVAL_OPTIONS,
        index=0,
        help="Controls how context is retrieved.",
    )
    scope = st.selectbox(
        "Document scope",
        SCOPE_OPTIONS,
        index=0,
        help=(
            "project: internal docs only, all: include external PDFs, external: external PDFs only."
        ),
    )
    k = st.number_input(
        "Top-k",
        min_value=1,
        max_value=30,
        value=6,
        step=1,
        help="How many chunks the retriever should return.",
    )
    q = st.text_area(
        "Request",
        value="Create a solution outline including architecture, risks, and assumptions.",
        height=140,
    )
    if st.button("Run Agent"):
        headers = {
            "X-Retrieval-Mode": mode,
            "X-Doc-Scope": scope,
            "X-Retrieval-K": str(k),
        }
        r = requests.post(
            f"{API_URL}/agent",
            json={"question": q},
            headers=headers,
            timeout=300,
        )
        payload = r.json()
        _render_response(payload)

st.divider()

use_structured = st.checkbox("Use structured inputs", value=False)
if use_structured:
    with st.expander("Structured request (optional)", expanded=True):
        st.caption("Use this to provide structured inputs while still calling the agent endpoint.")
        use_case = st.selectbox(
            "Use case",
            ["ADR", "Solution Outline", "User Stories", "Risk Assessment"],
        )
        context_query = st.text_input(
            "Context query (what to retrieve)",
            value="architecture notes, runbook, security policy",
        )

        if use_case == "ADR":
            decision = st.text_input(
                "Decision",
                value="Use Qdrant as vector database for RAG retrieval",
            )
            alternatives = st.text_input(
                "Alternatives (comma separated)",
                value="FAISS, Chroma, Elasticsearch",
            )
            structured_prompt = (
                "Create an ADR.\n"
                f"Decision: {decision}\n"
                f"Alternatives: {alternatives}\n"
                f"Context query: {context_query}\n"
            )
        elif use_case == "Solution Outline":
            request = st.text_area(
                "Request",
                value="Create a solution outline including architecture, risks, and assumptions.",
                height=100,
            )
            structured_prompt = (
                f"Create a solution outline.\nRequest: {request}\nContext query: {context_query}\n"
            )
        elif use_case == "User Stories":
            request = st.text_area(
                "Request",
                value="Create user stories and acceptance criteria for incident handling.",
                height=100,
            )
            structured_prompt = (
                "Create user stories and acceptance criteria.\n"
                f"Request: {request}\n"
                f"Context query: {context_query}\n"
            )
        else:
            request = st.text_area(
                "Request",
                value="Create a risk assessment and mitigations for this project.",
                height=100,
            )
            structured_prompt = (
                f"Create a risk assessment.\nRequest: {request}\nContext query: {context_query}\n"
            )

    with st.expander("Show generated prompt", expanded=False):
        st.code(structured_prompt)
        st.caption("Tip: use the copy icon in the code block to copy the prompt.")

    if st.button("Run Structured Request"):
        headers = {
            "X-Retrieval-Mode": mode,
            "X-Doc-Scope": scope,
            "X-Retrieval-K": str(k),
        }
        r = requests.post(
            f"{API_URL}/agent",
            json={"question": structured_prompt},
            headers=headers,
            timeout=300,
        )
        payload = r.json()
        _render_response(payload)

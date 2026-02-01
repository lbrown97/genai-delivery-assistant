import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")
RETRIEVAL_OPTIONS = ["mmr", "hybrid", "similarity"]

st.set_page_config(page_title="GenAI Delivery Assistant", layout="wide")
st.title("GenAI Delivery Assistant (Agentic)")

col1, col2 = st.columns([1, 2])

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
    q = st.text_area(
        "Request",
        value="Create a solution outline including architecture, risks, and assumptions.",
        height=140,
    )
    if st.button("Run Agent"):
        r = requests.post(
            f"{API_URL}/agent",
            json={"question": q},
            headers={"X-Retrieval-Mode": mode},
            timeout=300,
        )
        payload = r.json()
        if isinstance(payload, dict) and payload.get("agent_tool"):
            st.caption(f"Selected tool: {payload.get('agent_tool')}")
            if payload.get("agent_args"):
                st.code(payload.get("agent_args"), language="json")
        st.json(payload)

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
            "Create a solution outline.\n"
            f"Request: {request}\n"
            f"Context query: {context_query}\n"
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
            "Create a risk assessment.\n"
            f"Request: {request}\n"
            f"Context query: {context_query}\n"
        )

    with st.expander("Show generated prompt", expanded=False):
        st.code(structured_prompt)
        st.caption("Tip: use the copy icon in the code block to copy the prompt.")

    if st.button("Run Structured Request"):
        r = requests.post(
            f"{API_URL}/agent",
            json={"question": structured_prompt},
            headers={"X-Retrieval-Mode": mode},
            timeout=300,
        )
        payload = r.json()
        if isinstance(payload, dict) and payload.get("agent_tool"):
            st.caption(f"Selected tool: {payload.get('agent_tool')}")
            if payload.get("agent_args"):
                st.code(payload.get("agent_args"), language="json")
        st.json(payload)

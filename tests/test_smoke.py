def test_router_prompt_loads():
    # Ensure router prompt is available and not empty
    from app.agent.router_prompts import _read_prompt

    content = _read_prompt("tool_router.md")
    assert "tool" in content.lower()


def test_guardrails_citation_check():
    from app.agent.guardrails import validate_citations

    assert validate_citations("Answer [source1]")
    assert not validate_citations("No citations here")

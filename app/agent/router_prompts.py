from pathlib import Path

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

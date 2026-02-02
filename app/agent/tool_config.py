from app.agent.router_prompts import ADR_PROMPT, RISK_PROMPT, SOLUTION_PROMPT, STORIES_PROMPT
from app.agent.schemas import ADR, RiskAssessment, SolutionOutline, UserStories

TOOL_CONFIG = {
    "adr": {
        "prompt": ADR_PROMPT,
        "model_cls": ADR,
        "temperature": 0.1,
        "tags": ["use_case:adr"],
        "gate": {"min_docs": 2, "min_score": 0.8},
        "error_message": "Not enough context from documents for ADR.",
    },
    "solution_outline": {
        "prompt": SOLUTION_PROMPT,
        "model_cls": SolutionOutline,
        "temperature": 0.2,
        "tags": ["use_case:solution_outline"],
        "gate": {"min_docs": 2, "min_score": 0.8},
        "error_message": "Not enough context from documents for Solution Outline.",
    },
    "user_stories": {
        "prompt": STORIES_PROMPT,
        "model_cls": UserStories,
        "temperature": 0.2,
        "tags": ["use_case:user_stories"],
        "gate": {"min_docs": 1, "min_score": 0.8},
        "error_message": "Not enough context from documents for User Stories.",
    },
    "risk_assessment": {
        "prompt": RISK_PROMPT,
        "model_cls": RiskAssessment,
        "temperature": 0.2,
        "tags": ["use_case:risk_assessment"],
        "gate": {"min_docs": 1, "min_score": 0.8},
        "error_message": "Not enough context from documents for Risk Assessment.",
    },
}

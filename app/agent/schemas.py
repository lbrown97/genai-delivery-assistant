from typing import List

from pydantic import BaseModel, Field


class ADR(BaseModel):
    """Structured schema for Architecture Decision Records."""

    title: str = Field(..., min_length=5)
    context: str
    decision: str
    alternatives: List[str]
    consequences: List[str]
    sources: List[str]  # source_ids


class SolutionOutline(BaseModel):
    """Structured schema for solution-outline responses."""

    title: str = Field(..., min_length=5)
    summary: str
    architecture: List[str]
    risks: List[str]
    assumptions: List[str]
    sources: List[str]


class UserStories(BaseModel):
    """Structured schema for user stories and acceptance criteria."""

    epic: str
    stories: List[str]
    acceptance_criteria: List[str]
    sources: List[str]


class RiskAssessment(BaseModel):
    """Structured schema for risk-assessment responses."""

    risks: List[str]
    mitigations: List[str]
    sources: List[str]


class ToolCall(BaseModel):
    """Schema for router-selected tool calls."""

    tool: str = Field(
        ...,
        description="One of: ask, adr, solution_outline, user_stories, risk_assessment",
    )
    args: dict

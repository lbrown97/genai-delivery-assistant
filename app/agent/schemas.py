from pydantic import BaseModel, Field
from typing import List

class ADR(BaseModel):
    title: str = Field(..., min_length=5)
    context: str
    decision: str
    alternatives: List[str]
    consequences: List[str]
    sources: List[str]  # source_ids

class SolutionOutline(BaseModel):
    title: str = Field(..., min_length=5)
    summary: str
    architecture: List[str]
    risks: List[str]
    assumptions: List[str]
    sources: List[str]

class UserStories(BaseModel):
    epic: str
    stories: List[str]
    acceptance_criteria: List[str]
    sources: List[str]

class RiskAssessment(BaseModel):
    risks: List[str]
    mitigations: List[str]
    sources: List[str]

class ToolCall(BaseModel):
    tool: str = Field(..., description="One of: ask, adr, solution_outline, user_stories, risk_assessment")
    args: dict

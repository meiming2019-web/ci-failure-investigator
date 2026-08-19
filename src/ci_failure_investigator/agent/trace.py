from pydantic import BaseModel, ConfigDict, Field

from ci_failure_investigator.agent.actions import InvestigationDecision


class InvestigationTraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(strict=True, ge=1)
    decision: InvestigationDecision
    evidence_id: str | None = None

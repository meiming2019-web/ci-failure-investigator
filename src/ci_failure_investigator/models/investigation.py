from pydantic import BaseModel, ConfigDict, Field, model_validator

from ci_failure_investigator.models.evidence import Evidence
from ci_failure_investigator.models.failure import FailureUnderstanding
from ci_failure_investigator.models.hypothesis import Hypothesis


class InvestigationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure: FailureUnderstanding
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    tool_calls_used: int = Field(default=0, strict=True, ge=0)
    tool_call_budget: int = Field(default=8, strict=True, ge=1)

    @model_validator(mode="after")
    def validate_investigation_invariants(self) -> "InvestigationState":
        evidence_ids = [item.id for item in self.evidence]
        duplicate_evidence_ids = _duplicates(evidence_ids)
        if duplicate_evidence_ids:
            raise ValueError(f"duplicate evidence IDs: {sorted(duplicate_evidence_ids)}")

        hypothesis_ids = [item.id for item in self.hypotheses]
        duplicate_hypothesis_ids = _duplicates(hypothesis_ids)
        if duplicate_hypothesis_ids:
            raise ValueError(f"duplicate hypothesis IDs: {sorted(duplicate_hypothesis_ids)}")

        available_evidence_ids = set(evidence_ids)
        for hypothesis in self.hypotheses:
            referenced_ids = set(hypothesis.supporting_evidence_ids) | set(
                hypothesis.contradicting_evidence_ids
            )
            missing_ids = referenced_ids - available_evidence_ids
            if missing_ids:
                raise ValueError(
                    f"hypothesis {hypothesis.id!r} references missing evidence IDs: "
                    f"{sorted(missing_ids)}"
                )

        if self.tool_calls_used > self.tool_call_budget:
            raise ValueError(
                "tool_calls_used cannot exceed tool_call_budget: "
                f"{self.tool_calls_used} > {self.tool_call_budget}"
            )
        return self


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates

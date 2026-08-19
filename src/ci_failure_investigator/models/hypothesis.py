from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HypothesisStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: HypothesisStatus = HypothesisStatus.UNVERIFIED
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    revision_reason: str | None = None

    @model_validator(mode="after")
    def evidence_ids_do_not_overlap(self) -> "Hypothesis":
        overlap = set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids)
        if overlap:
            raise ValueError(f"evidence IDs cannot be both supporting and contradicting: {sorted(overlap)}")
        return self

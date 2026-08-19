from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ci_failure_investigator.models.hypothesis import Hypothesis


class ListAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["LIST"] = "LIST"
    path: str = "."


class SearchAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["SEARCH"] = "SEARCH"
    query: str = Field(min_length=1)
    file_glob: str | None = None


class ReadAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["READ"] = "READ"
    path: str
    start_line: int = Field(strict=True, ge=1)
    end_line: int = Field(strict=True, ge=1)

    @model_validator(mode="after")
    def validate_line_span(self) -> "ReadAction":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be at least start_line")
        if self.end_line - self.start_line + 1 > 200:
            raise ValueError("READ range cannot exceed 200 lines")
        return self


class ConcludeAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["CONCLUDE"] = "CONCLUDE"


Action = Annotated[
    ListAction | SearchAction | ReadAction | ConcludeAction,
    Field(discriminator="action_type"),
]


class InvestigationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    hypotheses: list[Hypothesis] | None = None

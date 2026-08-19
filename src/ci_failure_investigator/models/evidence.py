from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    step_number: int = Field(strict=True, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

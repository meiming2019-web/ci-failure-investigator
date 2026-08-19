from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FailureCategory(str, Enum):
    TEST_FAILURE = "TEST_FAILURE"
    COLLECTION_ERROR = "COLLECTION_ERROR"
    IMPORT_ERROR = "IMPORT_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    UNKNOWN = "UNKNOWN"


class TracebackFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    line_number: int = Field(strict=True, ge=1)
    function: str | None = None


class FailureUnderstanding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_category: FailureCategory = FailureCategory.UNKNOWN
    failing_test: str | None = None
    exception_type: str | None = None
    error_message: str | None = None
    traceback_frames: list[TracebackFrame] = Field(default_factory=list)
    implicated_paths: list[str] = Field(default_factory=list)
    raw_excerpt: str = ""
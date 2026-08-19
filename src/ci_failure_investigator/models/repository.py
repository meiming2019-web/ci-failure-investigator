from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RepositoryEntryType(str, Enum):
    FILE = "FILE"
    DIRECTORY = "DIRECTORY"


class RepositoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    entry_type: RepositoryEntryType


class RepositoryListResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    entries: list[RepositoryEntry] = Field(default_factory=list)
    truncated: bool = False


class RepositorySearchMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    line_number: int = Field(strict=True, ge=1)
    line_text: str


class RepositorySearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    matches: list[RepositorySearchMatch] = Field(default_factory=list)
    truncated: bool = False


class RepositoryReadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    start_line: int = Field(strict=True, ge=1)
    end_line: int = Field(strict=True, ge=1)
    content: str
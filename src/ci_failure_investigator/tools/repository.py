import fnmatch
import os
from pathlib import Path

from ci_failure_investigator.models import (
    RepositoryEntry,
    RepositoryEntryType,
    RepositoryListResult,
    RepositoryReadResult,
    RepositorySearchMatch,
    RepositorySearchResult,
)

MAX_LIST_ENTRIES = 200
MAX_SEARCH_MATCHES = 50
MAX_READ_LINES = 200


class RepositoryToolError(ValueError):
    """Raised when a repository observation request is invalid or unsafe."""


def _repository_root(repo_root: str | os.PathLike[str]) -> Path:
    root = Path(repo_root).resolve()
    if not root.exists() or not root.is_dir():
        raise RepositoryToolError("repository root must exist and be a directory")
    return root


def _safe_path(root: Path, relative_path: str) -> Path:
    requested = Path(relative_path)
    if requested.is_absolute():
        raise RepositoryToolError("repository paths must be relative")
    target = (root / requested).resolve()
    if not target.is_relative_to(root):
        raise RepositoryToolError("repository path escapes the repository root")
    return target


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix() or "."


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:4096]


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
        if _is_binary(data):
            return None
        return data.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def list_repository_path(
    repo_root: str | os.PathLike[str], relative_path: str = "."
) -> RepositoryListResult:
    root = _repository_root(repo_root)
    target = _safe_path(root, relative_path)
    if not target.exists() or not target.is_dir():
        raise RepositoryToolError("LIST path must be an existing directory")

    children: list[tuple[str, Path]] = []
    for child in target.iterdir():
        try:
            resolved_child = child.resolve()
        except OSError as error:
            raise RepositoryToolError("unable to resolve LIST entry") from error
        if not resolved_child.is_relative_to(root):
            continue
        children.append((_relative_path(root, child), child))

    children.sort(key=lambda item: item[0])
    truncated = len(children) > MAX_LIST_ENTRIES
    entries = []
    for path, child in children[:MAX_LIST_ENTRIES]:
        if child.is_dir():
            entry_type = RepositoryEntryType.DIRECTORY
        elif child.is_file():
            entry_type = RepositoryEntryType.FILE
        else:
            continue
        entries.append(RepositoryEntry(path=path, entry_type=entry_type))

    return RepositoryListResult(
        path=_relative_path(root, target), entries=entries, truncated=truncated
    )


def search_repository(
    repo_root: str | os.PathLike[str], query: str, file_glob: str | None = None
) -> RepositorySearchResult:
    if not query:
        raise RepositoryToolError("search query must be non-empty")
    root = _repository_root(repo_root)
    matches: list[RepositorySearchMatch] = []
    truncated = False

    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name != ".git" and not (current / name).is_symlink()
        )
        for name in sorted(file_names):
            path = current / name
            relative = _relative_path(root, path)
            if file_glob and not (
                fnmatch.fnmatch(relative, file_glob) or fnmatch.fnmatch(path.name, file_glob)
            ):
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if not resolved.is_relative_to(root) or not resolved.is_file():
                continue
            text = _read_text(resolved)
            if text is None:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        truncated = True
                        break
                    matches.append(
                        RepositorySearchMatch(
                            path=relative, line_number=line_number, line_text=line
                        )
                    )
            if truncated:
                break
        if truncated:
            break

    return RepositorySearchResult(query=query, matches=matches, truncated=truncated)


def read_repository_file(
    repo_root: str | os.PathLike[str],
    relative_path: str,
    start_line: int,
    end_line: int,
) -> RepositoryReadResult:
    if start_line < 1:
        raise RepositoryToolError("start_line must be at least 1")
    if end_line < start_line:
        raise RepositoryToolError("end_line must be at least start_line")
    if end_line - start_line + 1 > MAX_READ_LINES:
        raise RepositoryToolError("READ range cannot exceed 200 lines")

    root = _repository_root(repo_root)
    target = _safe_path(root, relative_path)
    if not target.exists():
        raise RepositoryToolError("READ path does not exist")
    if not target.is_file():
        raise RepositoryToolError("READ path must be a regular file")

    text = _read_text(target)
    if text is None:
        raise RepositoryToolError("READ target must be a UTF-8 text file")
    lines = text.splitlines(keepends=True)
    if start_line > len(lines):
        raise RepositoryToolError("start_line is beyond end of file")

    actual_end_line = min(end_line, len(lines))
    content = "".join(lines[start_line - 1 : actual_end_line])
    return RepositoryReadResult(
        path=_relative_path(root, target),
        start_line=start_line,
        end_line=actual_end_line,
        content=content,
    )
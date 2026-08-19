import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from ci_failure_investigator.models import (
    RepositoryEntryType,
    RepositoryListResult,
    RepositoryReadResult,
    RepositorySearchMatch,
    RepositorySearchResult,
)
from ci_failure_investigator.tools import (
    RepositoryToolError,
    list_repository_path,
    read_repository_file,
    search_repository,
)


def make_repository(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    (tmp_path / "src" / "app.py").write_text("first\nneedle here\nlast\n", encoding="utf-8")
    (tmp_path / "src" / "notes.txt").write_text("needle in text\n", encoding="utf-8")
    (tmp_path / "tests" / "test_app.py").write_text("needle in test\n", encoding="utf-8")
    return tmp_path


def make_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable: {error}")


def test_list_root_and_nested_directory_is_immediate_and_sorted(tmp_path: Path) -> None:
    root = make_repository(tmp_path)

    root_result = list_repository_path(root, ".")
    nested_result = list_repository_path(root, "src")

    assert root_result.path == "."
    assert [(entry.path, entry.entry_type) for entry in root_result.entries] == [
        ("README.md", RepositoryEntryType.FILE),
        ("src", RepositoryEntryType.DIRECTORY),
        ("tests", RepositoryEntryType.DIRECTORY),
    ]
    assert [(entry.path, entry.entry_type) for entry in nested_result.entries] == [
        ("src/app.py", RepositoryEntryType.FILE),
        ("src/notes.txt", RepositoryEntryType.FILE),
    ]
    assert nested_result.truncated is False


def test_list_is_truncated_at_200_entries(tmp_path: Path) -> None:
    root = make_repository(tmp_path)
    for index in range(201):
        (root / f"file-{index:03d}.txt").write_text("", encoding="utf-8")

    result = list_repository_path(root)

    assert len(result.entries) == 200
    assert result.truncated is True
    assert result.entries[0].path == "README.md"


def test_read_inclusive_lines_and_eof_end_line(tmp_path: Path) -> None:
    root = make_repository(tmp_path)

    result = read_repository_file(root, "src/app.py", start_line=2, end_line=10)

    assert result == RepositoryReadResult(
        path="src/app.py",
        start_line=2,
        end_line=3,
        content="needle here\nlast\n",
    )


def test_read_rejects_invalid_ranges_and_targets(tmp_path: Path) -> None:
    root = make_repository(tmp_path)
    (root / "binary.bin").write_bytes(b"header\x00binary")
    cases = [
        ("src/app.py", 0, 1),
        ("src/app.py", 2, 1),
        ("src/app.py", 1, 201),
        ("src/app.py", 10, 10),
        ("missing.py", 1, 1),
        ("src", 1, 1),
        ("binary.bin", 1, 1),
        ("../outside.py", 1, 1),
        (os.path.abspath(root / "src" / "app.py"), 1, 1),
    ]

    for path, start_line, end_line in cases:
        with pytest.raises(RepositoryToolError):
            read_repository_file(root, path, start_line, end_line)


def test_search_is_literal_case_sensitive_ordered_and_glob_filtered(tmp_path: Path) -> None:
    root = make_repository(tmp_path)

    result = search_repository(root, "needle")
    python_result = search_repository(root, "needle", file_glob="*.py")
    uppercase_result = search_repository(root, "NEEDLE")

    assert [(match.path, match.line_number, match.line_text) for match in result.matches] == [
        ("src/app.py", 2, "needle here"),
        ("src/notes.txt", 1, "needle in text"),
        ("tests/test_app.py", 1, "needle in test"),
    ]
    assert [match.path for match in python_result.matches] == ["src/app.py", "tests/test_app.py"]
    assert uppercase_result.matches == []
    assert result.truncated is False


def test_search_is_bounded_at_50_matches_and_skips_git_and_binary(tmp_path: Path) -> None:
    root = make_repository(tmp_path)
    (root / ".git").mkdir()
    (root / ".git" / "hidden.txt").write_text("needle\n", encoding="utf-8")
    (root / "binary.bin").write_bytes(b"needle\x00not text")
    for index in range(51):
        (root / f"match-{index:02d}.txt").write_text("needle\n", encoding="utf-8")

    result = search_repository(root, "needle")

    assert len(result.matches) == 50
    assert result.truncated is True
    assert all(match.path != ".git/hidden.txt" for match in result.matches)
    assert all(match.path != "binary.bin" for match in result.matches)


def test_search_rejects_empty_query(tmp_path: Path) -> None:
    with pytest.raises(RepositoryToolError):
        search_repository(tmp_path, "")


def test_invalid_repository_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RepositoryToolError):
        list_repository_path(tmp_path / "missing")

    file_root = tmp_path / "root.txt"
    file_root.write_text("not a repository", encoding="utf-8")
    with pytest.raises(RepositoryToolError):
        list_repository_path(file_root)


def test_symlink_escape_is_rejected_or_skipped(tmp_path: Path) -> None:
    (tmp_path / "repo").mkdir()
    root = make_repository(tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret needle\n", encoding="utf-8")
    make_symlink(root / "escape", outside)

    with pytest.raises(RepositoryToolError):
        read_repository_file(root, "escape/secret.txt", 1, 1)
    with pytest.raises(RepositoryToolError):
        list_repository_path(root, "escape")

    search_result = search_repository(root, "secret")
    assert search_result.matches == []


def test_repository_models_reject_unknown_fields_and_public_imports() -> None:
    with pytest.raises(ValidationError):
        RepositoryListResult.model_validate(
            {"path": ".", "entries": [{"path": "x", "entry_type": "OTHER"}]}
        )
    with pytest.raises(ValidationError):
        RepositorySearchResult.model_validate({"query": "x", "unexpected": True})
    with pytest.raises(ValidationError):
        RepositorySearchMatch.model_validate(
            {"path": "a.py", "line_number": 0, "line_text": "x"}
        )

    assert RepositoryListResult.__name__ == "RepositoryListResult"
    assert RepositoryReadResult.__name__ == "RepositoryReadResult"
    assert RepositorySearchMatch.__name__ == "RepositorySearchMatch"
    assert RepositorySearchResult.__name__ == "RepositorySearchResult"

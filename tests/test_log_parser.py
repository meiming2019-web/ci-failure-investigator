import pytest
from pydantic import ValidationError

from ci_failure_investigator.logs import parse_ci_failure
from ci_failure_investigator.models import (
    FailureCategory,
    FailureUnderstanding,
    TracebackFrame,
)

BASIC_PYTEST_FAILURE = """\
=================================== FAILURES ===================================
____________________________ test_returns_error ____________________________

    def test_returns_error():
>       assert 1 == 2
E       assert 1 == 2

tests/test_api.py:12: AssertionError
=========================== short test summary info ============================
FAILED tests/test_api.py::test_returns_error - assert 1 == 2
"""


PYTHON_TRACEBACK = """\
Traceback (most recent call last):
  File "/workspace/project/src/app.py", line 42, in process_request
    return convert(value)
  File "/workspace/project/src/converter.py", line 18, in convert
    raise TypeError("expected str")
TypeError: expected str
"""


def test_parse_basic_pytest_failure() -> None:
    result = parse_ci_failure(BASIC_PYTEST_FAILURE)

    assert result.failure_category is FailureCategory.TEST_FAILURE
    assert result.failing_test == "tests/test_api.py::test_returns_error"
    assert "tests/test_api.py" in result.implicated_paths
    assert len(result.raw_excerpt.splitlines()) <= 30
    assert result.raw_excerpt.splitlines()[-1].startswith("FAILED ")


def test_parse_python_traceback() -> None:
    result = parse_ci_failure(PYTHON_TRACEBACK)

    assert result.failure_category is FailureCategory.RUNTIME_ERROR
    assert result.exception_type == "TypeError"
    assert result.error_message == "expected str"
    assert [(frame.path, frame.line_number, frame.function) for frame in result.traceback_frames] == [
        ("/workspace/project/src/app.py", 42, "process_request"),
        ("/workspace/project/src/converter.py", 18, "convert"),
    ]
    assert result.implicated_paths == [
        "/workspace/project/src/app.py",
        "/workspace/project/src/converter.py",
    ]


def test_parse_import_error() -> None:
    result = parse_ci_failure(
        """\
Traceback (most recent call last):
  File "/workspace/project/app.py", line 3, in <module>
    import missing_package
ModuleNotFoundError: No module named 'missing_package'
"""
    )

    assert result.failure_category is FailureCategory.IMPORT_ERROR
    assert result.exception_type == "ModuleNotFoundError"
    assert result.error_message == "No module named 'missing_package'"


def test_collection_error_takes_precedence_over_import_error() -> None:
    result = parse_ci_failure(
        """\
ERROR collecting tests/test_example.py
ImportError while importing test module
ImportError: cannot import name 'value'
"""
    )

    assert result.failure_category is FailureCategory.COLLECTION_ERROR
    assert result.implicated_paths == ["tests/test_example.py"]


def test_parse_pytest_failed_line_with_class_and_case() -> None:
    result = parse_ci_failure("FAILED tests/test_example.py::TestThing::test_case - AssertionError")

    assert result.failure_category is FailureCategory.TEST_FAILURE
    assert result.failing_test == "tests/test_example.py::TestThing::test_case"


def test_multiple_failed_tests_selects_last_node_id() -> None:
    result = parse_ci_failure(
        """\
FAILED tests/test_first.py::test_first - AssertionError
FAILED tests/test_last.py::TestThing::test_last - RuntimeError
"""
    )

    assert result.failing_test == "tests/test_last.py::TestThing::test_last"


def test_import_error_category_survives_later_runtime_error() -> None:
    result = parse_ci_failure(
        """\
ModuleNotFoundError: No module named 'missing_package'
RuntimeError: later failure
"""
    )

    assert result.failure_category is FailureCategory.IMPORT_ERROR
    assert result.exception_type == "RuntimeError"
    assert result.error_message == "later failure"


def test_duplicate_paths_are_removed_in_first_seen_order() -> None:
    result = parse_ci_failure(
        """\
  File "/workspace/app.py", line 1, in first
  File "/workspace/app.py", line 2, in second
  File "/workspace/other.py", line 3, in third
  File "/workspace/app.py", line 4, in fourth
"""
    )

    assert result.implicated_paths == ["/workspace/app.py", "/workspace/other.py"]


def test_empty_input_returns_empty_unknown_understanding() -> None:
    result = parse_ci_failure("")

    assert result == FailureUnderstanding(
        failure_category=FailureCategory.UNKNOWN,
        failing_test=None,
        exception_type=None,
        error_message=None,
        traceback_frames=[],
        implicated_paths=[],
        raw_excerpt="",
    )


def test_unknown_input_does_not_raise() -> None:
    result = parse_ci_failure("Build terminated unexpectedly.")

    assert result.failure_category is FailureCategory.UNKNOWN
    assert result.failing_test is None
    assert result.exception_type is None
    assert result.error_message is None


def test_raw_excerpt_is_bounded_to_final_non_empty_lines() -> None:
    log_text = "\n".join([f"noise-{index}" for index in range(35)] + ["FAILED tests/test.py::test_case"])

    excerpt_lines = parse_ci_failure(log_text).raw_excerpt.splitlines()

    assert len(excerpt_lines) == 30
    assert excerpt_lines[0] == "noise-6"
    assert excerpt_lines[-1] == "FAILED tests/test.py::test_case"
    assert excerpt_lines == sorted(excerpt_lines, key=lambda line: int(line.split("-")[-1]) if line.startswith("noise-") else 35)


def test_traceback_frame_rejects_invalid_line_number() -> None:
    with pytest.raises(ValidationError):
        TracebackFrame(path="app.py", line_number=0)


def test_failure_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        FailureUnderstanding.model_validate({"unexpected": True})

    with pytest.raises(ValidationError):
        TracebackFrame.model_validate({"path": "app.py", "line_number": 1, "unexpected": True})


def test_failure_types_and_parser_are_publicly_importable() -> None:
    assert FailureCategory.UNKNOWN.value == "UNKNOWN"
    assert FailureUnderstanding.__name__ == "FailureUnderstanding"
    assert TracebackFrame.__name__ == "TracebackFrame"
    assert callable(parse_ci_failure)

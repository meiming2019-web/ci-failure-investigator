import re

from ci_failure_investigator.models import (
    FailureCategory,
    FailureUnderstanding,
    TracebackFrame,
)

_FAILED_TEST_PATTERN = re.compile(r"^\s*FAILED\s+(?P<test>\S+::\S+)(?:\s+-\s*(?P<message>.*))?\s*$")
_COLLECTION_ERROR_PATTERN = re.compile(r"^\s*ERROR collecting(?:\s+(?P<path>\S+))?\s*$")
_TRACEBACK_FRAME_PATTERN = re.compile(
    r'^\s*File ["\'](?P<path>.+?)["\'], line (?P<line>\d+)(?:, in (?P<function>.+))?\s*$'
)
_PYTEST_LOCATION_PATTERN = re.compile(
    r"^\s*(?P<path>\S+\.py):(?P<line>\d+):\s*(?P<detail>.*)\s*$"
)
_EXCEPTION_PATTERN = re.compile(
    r"^\s*(?:E\s+)?(?P<type>[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))"
    r"(?::\s*(?P<message>.*))?\s*$"
)


def _bounded_excerpt(lines: list[str]) -> str:
    non_empty_lines = [line for line in lines if line.strip()]
    return "\n".join(non_empty_lines[-30:])


def _add_path(paths: list[str], path: str) -> None:
    if path not in paths:
        paths.append(path)


def parse_ci_failure(log_text: str) -> FailureUnderstanding:
    lines = log_text.splitlines()
    failing_test: str | None = None
    exception_type: str | None = None
    error_message: str | None = None
    traceback_frames: list[TracebackFrame] = []
    implicated_paths: list[str] = []
    collection_error = False
    import_error_observed = False

    for line in lines:
        failed_test_match = _FAILED_TEST_PATTERN.match(line)
        if failed_test_match:
            failing_test = failed_test_match.group("test")
            _add_path(implicated_paths, failing_test.split("::", maxsplit=1)[0])

        collection_match = _COLLECTION_ERROR_PATTERN.match(line)
        if collection_match:
            collection_error = True
            collection_path = collection_match.group("path")
            if collection_path:
                _add_path(implicated_paths, collection_path)

        frame_match = _TRACEBACK_FRAME_PATTERN.match(line)
        if frame_match:
            path = frame_match.group("path")
            function = frame_match.group("function")
            traceback_frames.append(
                TracebackFrame(
                    path=path,
                    line_number=int(frame_match.group("line")),
                    function=function.strip() if function else None,
                )
            )
            _add_path(implicated_paths, path)

        location_match = _PYTEST_LOCATION_PATTERN.match(line)
        if location_match:
            _add_path(implicated_paths, location_match.group("path"))

        exception_match = _EXCEPTION_PATTERN.match(line)
        if exception_match:
            exception_type = exception_match.group("type")
            error_message = exception_match.group("message") or None
            if exception_type in {"ImportError", "ModuleNotFoundError"}:
                import_error_observed = True

    if collection_error:
        category = FailureCategory.COLLECTION_ERROR
    elif import_error_observed:
        category = FailureCategory.IMPORT_ERROR
    elif failing_test is not None:
        category = FailureCategory.TEST_FAILURE
    elif exception_type is not None:
        category = FailureCategory.RUNTIME_ERROR
    else:
        category = FailureCategory.UNKNOWN

    return FailureUnderstanding(
        failure_category=category,
        failing_test=failing_test,
        exception_type=exception_type,
        error_message=error_message,
        traceback_frames=traceback_frames,
        implicated_paths=implicated_paths,
        raw_excerpt=_bounded_excerpt(lines),
    )
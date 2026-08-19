from pathlib import Path
from typing import Any, Literal

from ci_failure_investigator.agent import (
    ConcludeAction,
    InvestigationDecision,
    InvestigationTraceStep,
    ListAction,
    ReadAction,
    SearchAction,
    TerminationReason,
    run_investigation,
    run_smoke_investigation,
)
from ci_failure_investigator.agent.openai_policy import _OpenAIInvestigationDecision
from ci_failure_investigator.models import Evidence, FailureUnderstanding, InvestigationState


class FakeResponse:
    def __init__(self, decision: _OpenAIInvestigationDecision) -> None:
        self.output_parsed = decision


class FakeResponses:
    def __init__(self, decisions: list[_OpenAIInvestigationDecision]) -> None:
        self.responses = iter(FakeResponse(decision) for decision in decisions)
        self.calls: list[dict[str, Any]] = []

    def parse(
        self,
        *,
        model: str,
        input: str,
        instructions: str,
        text_format: type[_OpenAIInvestigationDecision],
    ) -> FakeResponse:
        self.calls.append(
            {
                "model": model,
                "input": input,
                "instructions": instructions,
                "text_format": text_format,
            }
        )
        return next(self.responses)


class FakeClient:
    def __init__(self, decisions: list[_OpenAIInvestigationDecision]) -> None:
        self.responses = FakeResponses(decisions)


def make_transport(
    action_type: Literal["LIST", "SEARCH", "READ", "CONCLUDE"],
    *,
    path: str | None = None,
    query: str | None = None,
    file_glob: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> _OpenAIInvestigationDecision:
    return _OpenAIInvestigationDecision(
        action_type=action_type,
        path=path,
        query=query,
        file_glob=file_glob,
        start_line=start_line,
        end_line=end_line,
        hypotheses=None,
    )


def make_state(
    *, evidence: list[Evidence] | None = None, tool_call_budget: int = 8
) -> InvestigationState:
    return InvestigationState(
        failure=FailureUnderstanding(),
        evidence=evidence or [],
        tool_call_budget=tool_call_budget,
    )


def make_evidence(evidence_id: str) -> Evidence:
    return Evidence(
        id=evidence_id,
        source_type="log",
        source_location="build.log:1",
        observation="Observed failure.",
        step_number=1,
    )


def test_immediate_conclude_has_one_trace_step(tmp_path: Path) -> None:
    result = run_investigation(
        make_state(),
        tmp_path,
        lambda state: InvestigationDecision(action=ConcludeAction()),
    )

    assert len(result.trace) == 1
    assert result.trace[0].iteration == 1
    assert result.trace[0].decision.action.action_type == "CONCLUDE"
    assert result.trace[0].evidence_id is None
    assert result.termination_reason is TerminationReason.CONCLUDED


def test_read_and_conclude_trace_references_actual_evidence(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("content\n", encoding="utf-8")
    decisions = iter(
        [
            InvestigationDecision(action=ReadAction(path="app.py", start_line=1, end_line=1)),
            InvestigationDecision(action=ConcludeAction()),
        ]
    )

    result = run_investigation(make_state(), tmp_path, lambda state: next(decisions))

    assert [step.iteration for step in result.trace] == [1, 2]
    assert result.trace[0].evidence_id == "E1"
    assert result.trace[1].decision.action.action_type == "CONCLUDE"
    assert result.trace[1].evidence_id is None
    assert result.state.evidence[0].id == result.trace[0].evidence_id


def test_multi_tool_trace_preserves_order(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("needle\n", encoding="utf-8")
    decisions = iter(
        [
            InvestigationDecision(action=ListAction()),
            InvestigationDecision(action=SearchAction(query="needle")),
            InvestigationDecision(action=ReadAction(path="app.py", start_line=1, end_line=1)),
            InvestigationDecision(action=ConcludeAction()),
        ]
    )

    result = run_investigation(make_state(), tmp_path, lambda state: next(decisions))

    assert [step.iteration for step in result.trace] == [1, 2, 3, 4]
    assert [step.decision.action.action_type for step in result.trace] == [
        "LIST",
        "SEARCH",
        "READ",
        "CONCLUDE",
    ]
    assert [step.evidence_id for step in result.trace] == ["E1", "E2", "E3", None]


def test_trace_uses_collision_free_evidence_id(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("content\n", encoding="utf-8")
    state = make_state(evidence=[make_evidence("E1"), make_evidence("E3")])
    decisions = iter(
        [
            InvestigationDecision(action=ReadAction(path="app.py", start_line=1, end_line=1)),
            InvestigationDecision(action=ConcludeAction()),
        ]
    )

    result = run_investigation(
        state,
        tmp_path,
        lambda current: next(decisions),
    )

    assert result.trace[0].evidence_id == "E2"


def test_initial_budget_exhaustion_has_empty_trace(tmp_path: Path) -> None:
    state = make_state(tool_call_budget=1)
    state = InvestigationState.model_validate(
        {**state.model_dump(), "tool_calls_used": 1}
    )

    result = run_investigation(
        state,
        tmp_path,
        lambda current: InvestigationDecision(action=ConcludeAction()),
    )

    assert result.trace == []
    assert result.termination_reason is TerminationReason.BUDGET_EXHAUSTED


def test_budget_exhaustion_after_tools_has_no_fake_trace_step(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("content\n", encoding="utf-8")
    result = run_investigation(
        make_state(tool_call_budget=2),
        tmp_path,
        lambda current: InvestigationDecision(
            action=ReadAction(path="app.py", start_line=1, end_line=1)
        ),
    )

    assert len(result.trace) == 2
    assert [step.evidence_id for step in result.trace] == ["E1", "E2"]
    assert result.termination_reason is TerminationReason.BUDGET_EXHAUSTED


def test_smoke_runner_composes_log_policy_graph_and_tools(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("needle\n", encoding="utf-8")
    client = FakeClient(
        [
            make_transport("SEARCH", query="needle"),
            make_transport("READ", path="app.py", start_line=1, end_line=1),
            make_transport("CONCLUDE"),
        ]
    )

    result = run_smoke_investigation(
        ci_log_text="FAILED tests/test_app.py::test_failure - AssertionError",
        repo_root=tmp_path,
        client=client,
        model="test-model",
    )

    assert result.state.failure.failing_test == "tests/test_app.py::test_failure"
    assert result.state.tool_calls_used == 2
    assert [step.evidence_id for step in result.trace] == ["E1", "E2", None]
    assert result.termination_reason is TerminationReason.CONCLUDED
    assert len(client.responses.calls) == 3


def test_smoke_runner_propagates_configured_budget(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("content\n", encoding="utf-8")
    client = FakeClient(
        [make_transport("READ", path="app.py", start_line=1, end_line=1)]
    )

    result = run_smoke_investigation(
        ci_log_text="unknown log",
        repo_root=tmp_path,
        client=client,
        model="test-model",
        tool_call_budget=1,
    )

    assert result.state.tool_call_budget == 1
    assert result.state.tool_calls_used == 1
    assert result.termination_reason is TerminationReason.BUDGET_EXHAUSTED
    assert len(result.trace) == 1


def test_trace_model_is_public() -> None:
    step = InvestigationTraceStep(
        iteration=1,
        decision=InvestigationDecision(action=ConcludeAction()),
    )

    assert step.evidence_id is None

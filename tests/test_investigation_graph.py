from pathlib import Path

import pytest
from pydantic import ValidationError

from ci_failure_investigator.agent import (
    ConcludeAction,
    InvestigationDecision,
    ListAction,
    ReadAction,
    SearchAction,
    TerminationReason,
    run_investigation,
)
from ci_failure_investigator.models import (
    Evidence,
    FailureUnderstanding,
    Hypothesis,
    InvestigationState,
)


def make_state(
    *,
    evidence: list[Evidence] | None = None,
    hypotheses: list[Hypothesis] | None = None,
    tool_calls_used: int = 0,
    tool_call_budget: int = 8,
) -> InvestigationState:
    return InvestigationState(
        failure=FailureUnderstanding(),
        evidence=evidence or [],
        hypotheses=hypotheses or [],
        tool_calls_used=tool_calls_used,
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


def make_hypothesis(hypothesis_id: str, supporting: list[str] | None = None) -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        description="A testable explanation.",
        supporting_evidence_ids=supporting or [],
    )


def test_immediate_conclude_does_not_use_tools(tmp_path: Path) -> None:
    calls = 0

    def policy(state: InvestigationState) -> InvestigationDecision:
        nonlocal calls
        calls += 1
        return InvestigationDecision(action=ConcludeAction())

    result = run_investigation(make_state(), tmp_path, policy)

    assert result.termination_reason is TerminationReason.CONCLUDED
    assert result.state.tool_calls_used == 0
    assert result.state.evidence == []
    assert calls == 1


def test_policy_mutations_cannot_change_authoritative_state(tmp_path: Path) -> None:
    initial = make_state()

    def policy(state: InvestigationState) -> InvestigationDecision:
        state.tool_calls_used = 99
        state.evidence.append(make_evidence("MUTATED"))
        return InvestigationDecision(action=ConcludeAction())

    result = run_investigation(initial, tmp_path, policy)

    assert initial.tool_calls_used == 0
    assert initial.evidence == []
    assert result.state.tool_calls_used == 0
    assert result.state.evidence == []


def test_single_read_records_grounded_evidence(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "app.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    decisions = iter(
        [
            InvestigationDecision(action=ReadAction(path="src/app.py", start_line=2, end_line=10)),
            InvestigationDecision(action=ConcludeAction()),
        ]
    )

    result = run_investigation(make_state(), tmp_path, lambda state: next(decisions))
    evidence = result.state.evidence[0]

    assert result.termination_reason is TerminationReason.CONCLUDED
    assert result.state.tool_calls_used == 1
    assert evidence.id == "E1"
    assert evidence.step_number == 1
    assert evidence.source_type == "repository_read"
    assert evidence.source_location == "src/app.py:2-3"
    assert evidence.observation == "two\nthree\n"


def test_multi_tool_loop_records_each_operation(tmp_path: Path) -> None:
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

    assert result.termination_reason is TerminationReason.CONCLUDED
    assert result.state.tool_calls_used == 3
    assert [evidence.id for evidence in result.state.evidence] == ["E1", "E2", "E3"]
    assert [evidence.step_number for evidence in result.state.evidence] == [1, 2, 3]
    assert [evidence.source_type for evidence in result.state.evidence] == [
        "repository_list",
        "repository_search",
        "repository_read",
    ]


def test_evidence_id_uses_smallest_available_identifier(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("content\n", encoding="utf-8")
    existing = [make_evidence("E1"), make_evidence("E3")]
    decisions = iter(
        [
            InvestigationDecision(action=ReadAction(path="app.py", start_line=1, end_line=1)),
            InvestigationDecision(action=ConcludeAction()),
        ]
    )

    result = run_investigation(
        make_state(evidence=existing), tmp_path, lambda state: next(decisions)
    )

    assert result.state.evidence[-1].id == "E2"


def test_budget_exhaustion_does_not_call_policy_after_last_tool(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("content\n", encoding="utf-8")
    calls = 0

    def policy(state: InvestigationState) -> InvestigationDecision:
        nonlocal calls
        calls += 1
        return InvestigationDecision(action=ReadAction(path="app.py", start_line=1, end_line=1))

    result = run_investigation(make_state(tool_call_budget=2), tmp_path, policy)

    assert result.termination_reason is TerminationReason.BUDGET_EXHAUSTED
    assert result.state.tool_calls_used == 2
    assert len(result.state.evidence) == 2
    assert calls == 2


def test_initially_exhausted_budget_skips_policy(tmp_path: Path) -> None:
    calls = 0

    def policy(state: InvestigationState) -> InvestigationDecision:
        nonlocal calls
        calls += 1
        return InvestigationDecision(action=ConcludeAction())

    result = run_investigation(make_state(tool_calls_used=2, tool_call_budget=2), tmp_path, policy)

    assert result.termination_reason is TerminationReason.BUDGET_EXHAUSTED
    assert calls == 0


def test_hypothesis_update_is_revalidated_after_evidence_exists(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("content\n", encoding="utf-8")
    decisions = iter(
        [
            InvestigationDecision(
                action=ReadAction(path="app.py", start_line=1, end_line=1),
                hypotheses=[make_hypothesis("H1")],
            ),
            InvestigationDecision(
                action=ConcludeAction(),
                hypotheses=[make_hypothesis("H1", supporting=["E1"])],
            ),
        ]
    )

    result = run_investigation(make_state(), tmp_path, lambda state: next(decisions))

    assert result.state.hypotheses[0].supporting_evidence_ids == ["E1"]


def test_invalid_prospective_grounding_is_rejected_before_tool_execution(tmp_path: Path) -> None:
    calls = 0

    def policy(state: InvestigationState) -> InvestigationDecision:
        nonlocal calls
        calls += 1
        return InvestigationDecision(
            action=ReadAction(path="missing.py", start_line=1, end_line=1),
            hypotheses=[make_hypothesis("H1", supporting=["E1"])],
        )

    initial = make_state()
    with pytest.raises(ValidationError, match="H1.*E1"):
        run_investigation(initial, tmp_path, policy)

    assert calls == 1
    assert initial.tool_calls_used == 0
    assert initial.evidence == []
    assert initial.hypotheses == []


def test_repository_tool_error_becomes_evidence_and_allows_conclusion(tmp_path: Path) -> None:
    decisions = iter(
        [
            InvestigationDecision(
                action=ReadAction(path="missing.py", start_line=1, end_line=1),
                hypotheses=[make_hypothesis("H1")],
            ),
            InvestigationDecision(action=ConcludeAction()),
        ]
    )

    result = run_investigation(make_state(), tmp_path, lambda state: next(decisions))

    error_evidence = result.state.evidence[0]
    assert result.termination_reason is TerminationReason.CONCLUDED
    assert result.trace[0].decision.action.action_type == "READ"
    assert result.trace[0].evidence_id == "E1"
    assert error_evidence.source_type == "repository_tool_error"
    assert "READ path does not exist" in error_evidence.observation
    assert error_evidence.metadata == {
        "operation": "READ",
        "error_type": "RepositoryToolError",
        "path": "missing.py",
        "start_line": 1,
        "end_line": 1,
    }
    assert result.state.tool_calls_used == 1
    assert result.state.hypotheses == [make_hypothesis("H1")]


def test_repository_error_evidence_is_visible_to_next_policy_call(tmp_path: Path) -> None:
    seen_states: list[InvestigationState] = []
    decisions = iter(
        [
            InvestigationDecision(action=ReadAction(path="missing.py", start_line=1, end_line=1)),
            InvestigationDecision(action=ConcludeAction()),
        ]
    )

    def policy(state: InvestigationState) -> InvestigationDecision:
        seen_states.append(state)
        return next(decisions)

    run_investigation(make_state(), tmp_path, policy)

    assert len(seen_states) == 2
    assert seen_states[1].evidence[0].source_type == "repository_tool_error"
    assert seen_states[1].evidence[0].id == "E1"


def test_model_recovers_after_repository_tool_error(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("needle\n", encoding="utf-8")
    decisions = iter(
        [
            InvestigationDecision(action=ReadAction(path="missing.py", start_line=1, end_line=1)),
            InvestigationDecision(action=SearchAction(query="needle")),
            InvestigationDecision(action=ConcludeAction()),
        ]
    )

    result = run_investigation(make_state(), tmp_path, lambda state: next(decisions))

    assert result.termination_reason is TerminationReason.CONCLUDED
    assert [evidence.id for evidence in result.state.evidence] == ["E1", "E2"]
    assert result.state.evidence[0].source_type == "repository_tool_error"
    assert result.state.evidence[1].source_type == "repository_search"
    assert [step.evidence_id for step in result.trace] == ["E1", "E2", None]


def test_failed_repository_action_consumes_final_budget(tmp_path: Path) -> None:
    calls = 0

    def policy(state: InvestigationState) -> InvestigationDecision:
        nonlocal calls
        calls += 1
        return InvestigationDecision(action=ReadAction(path="missing.py", start_line=1, end_line=1))

    result = run_investigation(make_state(tool_call_budget=1), tmp_path, policy)

    assert calls == 1
    assert result.termination_reason is TerminationReason.BUDGET_EXHAUSTED
    assert result.state.tool_calls_used == 1
    assert result.state.evidence[0].source_type == "repository_tool_error"
    assert len(result.trace) == 1
    assert result.trace[0].evidence_id == "E1"


def test_repository_error_evidence_uses_smallest_available_identifier(tmp_path: Path) -> None:
    state = make_state(evidence=[make_evidence("E1"), make_evidence("E3")])
    decisions = iter(
        [
            InvestigationDecision(action=ReadAction(path="missing.py", start_line=1, end_line=1)),
            InvestigationDecision(action=ConcludeAction()),
        ]
    )

    result = run_investigation(state, tmp_path, lambda current: next(decisions))

    assert result.state.evidence[-1].id == "E2"
    assert result.trace[0].evidence_id == "E2"


def test_list_repository_tool_error_is_recoverable(tmp_path: Path) -> None:
    decisions = iter(
        [
            InvestigationDecision(action=ListAction(path="missing")),
            InvestigationDecision(action=ConcludeAction()),
        ]
    )

    result = run_investigation(make_state(), tmp_path, lambda state: next(decisions))

    assert result.termination_reason is TerminationReason.CONCLUDED
    assert result.state.evidence[0].source_type == "repository_tool_error"
    assert result.state.evidence[0].metadata == {
        "operation": "LIST",
        "error_type": "RepositoryToolError",
        "path": "missing",
    }


def test_unexpected_repository_exception_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ci_failure_investigator.agent.graph as graph_module

    def raise_unexpected(*args: object, **kwargs: object) -> object:
        raise RuntimeError("unexpected repository failure")

    monkeypatch.setattr(graph_module, "read_repository_file", raise_unexpected)

    with pytest.raises(RuntimeError, match="unexpected repository failure"):
        run_investigation(
            make_state(),
            tmp_path,
            lambda state: InvestigationDecision(
                action=ReadAction(path="app.py", start_line=1, end_line=1)
            ),
        )


def test_action_validation_rejects_invalid_ranges_and_queries() -> None:
    with pytest.raises(ValidationError):
        ReadAction(path="app.py", start_line=2, end_line=1)
    with pytest.raises(ValidationError):
        ReadAction(path="app.py", start_line=1, end_line=201)
    with pytest.raises(ValidationError):
        SearchAction(query="")


def test_agent_public_imports() -> None:
    assert ListAction(action_type="LIST").path == "."
    assert SearchAction(action_type="SEARCH", query="needle").query == "needle"

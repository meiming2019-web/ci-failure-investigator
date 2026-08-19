import pytest
from pydantic import ValidationError

from ci_failure_investigator.models import (
    Evidence,
    FailureUnderstanding,
    Hypothesis,
    InvestigationState,
)


def make_evidence(evidence_id: str) -> Evidence:
    return Evidence(
        id=evidence_id,
        source_type="log",
        source_location="build.log:1",
        observation="Observed failure output.",
        step_number=1,
    )


def make_hypothesis(
    hypothesis_id: str,
    supporting: list[str] | None = None,
    contradicting: list[str] | None = None,
) -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        description="The observed failure has a specific cause.",
        supporting_evidence_ids=supporting or [],
        contradicting_evidence_ids=contradicting or [],
    )


def test_state_defaults_from_required_failure() -> None:
    state = InvestigationState(failure=FailureUnderstanding())

    assert state.hypotheses == []
    assert state.evidence == []
    assert state.tool_calls_used == 0
    assert state.tool_call_budget == 8


def test_state_accepts_grounded_hypothesis() -> None:
    state = InvestigationState(
        failure=FailureUnderstanding(),
        evidence=[make_evidence("E1"), make_evidence("E2")],
        hypotheses=[make_hypothesis("H1", supporting=["E1"], contradicting=["E2"])],
    )

    assert state.hypotheses[0].id == "H1"


def test_state_rejects_duplicate_evidence_ids() -> None:
    with pytest.raises(ValidationError, match="E1"):
        InvestigationState(
            failure=FailureUnderstanding(),
            evidence=[make_evidence("E1"), make_evidence("E1")],
        )


def test_state_rejects_duplicate_hypothesis_ids() -> None:
    with pytest.raises(ValidationError, match="H1"):
        InvestigationState(
            failure=FailureUnderstanding(),
            hypotheses=[make_hypothesis("H1"), make_hypothesis("H1")],
        )


@pytest.mark.parametrize("reference_field", ["supporting", "contradicting"])
def test_state_rejects_missing_evidence_reference(reference_field: str) -> None:
    kwargs = {reference_field: ["E2"]}

    with pytest.raises(ValidationError, match="H1.*E2"):
        InvestigationState(
            failure=FailureUnderstanding(),
            evidence=[make_evidence("E1")],
            hypotheses=[make_hypothesis("H1", **kwargs)],
        )


def test_hypotheses_may_share_evidence() -> None:
    state = InvestigationState(
        failure=FailureUnderstanding(),
        evidence=[make_evidence("E1")],
        hypotheses=[
            make_hypothesis("H1", supporting=["E1"]),
            make_hypothesis("H2", contradicting=["E1"]),
        ],
    )

    assert len(state.hypotheses) == 2


@pytest.mark.parametrize(
    ("tool_calls_used", "tool_call_budget"),
    [(0, 8), (8, 8)],
)
def test_state_accepts_valid_tool_budgets(tool_calls_used: int, tool_call_budget: int) -> None:
    state = InvestigationState(
        failure=FailureUnderstanding(),
        tool_calls_used=tool_calls_used,
        tool_call_budget=tool_call_budget,
    )

    assert state.tool_calls_used == tool_calls_used


@pytest.mark.parametrize(
    ("tool_calls_used", "tool_call_budget"),
    [(9, 8), (0, 0), (-1, 8)],
)
def test_state_rejects_invalid_tool_budgets(
    tool_calls_used: int, tool_call_budget: int
) -> None:
    with pytest.raises(ValidationError):
        InvestigationState(
            failure=FailureUnderstanding(),
            tool_calls_used=tool_calls_used,
            tool_call_budget=tool_call_budget,
        )


def test_state_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        InvestigationState.model_validate({"failure": {}, "unexpected": True})


def test_investigation_state_public_import() -> None:
    assert InvestigationState.__name__ == "InvestigationState"
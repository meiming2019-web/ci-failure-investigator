import pytest
from pydantic import ValidationError

from ci_failure_investigator.models import Evidence, Hypothesis, HypothesisStatus


def test_evidence_can_be_constructed_with_metadata() -> None:
    evidence = Evidence(
        id="evidence-1",
        source_type="log",
        source_location="build.log:42",
        observation="The test process exited with code 1.",
        step_number=1,
        metadata={"exit_code": 1, "tags": ["test"]},
    )

    assert evidence.metadata == {"exit_code": 1, "tags": ["test"]}


def test_evidence_metadata_defaults_to_empty_dictionary() -> None:
    evidence = Evidence(
        id="evidence-1",
        source_type="log",
        source_location="build.log:42",
        observation="The test process exited with code 1.",
        step_number=1,
    )

    assert evidence.metadata == {}


def test_evidence_rejects_non_positive_step_number() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            id="evidence-1",
            source_type="log",
            source_location="build.log:42",
            observation="The test process exited with code 1.",
            step_number=0,
        )


@pytest.mark.parametrize(
    "field_name",
    ["id", "source_type", "source_location", "observation"],
)
def test_evidence_rejects_empty_required_fields(field_name: str) -> None:
    values = {
        "id": "evidence-1",
        "source_type": "log",
        "source_location": "build.log:42",
        "observation": "The test process exited with code 1.",
        "step_number": 1,
    }
    values[field_name] = ""

    with pytest.raises(ValidationError):
        Evidence.model_validate(values)


def test_hypothesis_defaults_to_unverified() -> None:
    hypothesis = Hypothesis(id="hypothesis-1", description="The dependency is incompatible.")

    assert hypothesis.status is HypothesisStatus.UNVERIFIED
    assert hypothesis.supporting_evidence_ids == []
    assert hypothesis.contradicting_evidence_ids == []
    assert hypothesis.revision_reason is None


@pytest.mark.parametrize("field_name", ["id", "description"])
def test_hypothesis_rejects_empty_required_fields(field_name: str) -> None:
    values = {
        "id": "hypothesis-1",
        "description": "The dependency is incompatible.",
    }
    values[field_name] = ""

    with pytest.raises(ValidationError):
        Hypothesis.model_validate(values)


def test_hypothesis_accepts_supporting_and_contradicting_evidence() -> None:
    hypothesis = Hypothesis(
        id="hypothesis-1",
        description="The dependency is incompatible.",
        status=HypothesisStatus.SUPPORTED,
        supporting_evidence_ids=["evidence-1"],
        contradicting_evidence_ids=["evidence-2"],
        revision_reason="The version constraint was confirmed.",
    )

    assert hypothesis.supporting_evidence_ids == ["evidence-1"]
    assert hypothesis.contradicting_evidence_ids == ["evidence-2"]


def test_hypothesis_rejects_overlapping_evidence_ids() -> None:
    with pytest.raises(ValidationError):
        Hypothesis(
            id="hypothesis-1",
            description="The dependency is incompatible.",
            supporting_evidence_ids=["evidence-1"],
            contradicting_evidence_ids=["evidence-1"],
        )


def test_evidence_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Evidence.model_validate(
            {
                "id": "evidence-1",
                "source_type": "log",
                "source_location": "build.log:42",
                "observation": "The test process exited with code 1.",
                "step_number": 1,
                "unexpected": True,
            }
        )


def test_hypothesis_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Hypothesis.model_validate(
            {
                "id": "hypothesis-1",
                "description": "The dependency is incompatible.",
                "unexpected": True,
            }
        )


def test_models_are_available_from_public_package() -> None:
    assert Evidence.__name__ == "Evidence"
    assert Hypothesis.__name__ == "Hypothesis"
    assert HypothesisStatus.UNVERIFIED.value == "UNVERIFIED"
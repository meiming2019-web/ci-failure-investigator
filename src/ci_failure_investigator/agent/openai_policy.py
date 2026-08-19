from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from ci_failure_investigator.agent.actions import (
    Action,
    ConcludeAction,
    InvestigationDecision,
    ListAction,
    ReadAction,
    SearchAction,
)
from ci_failure_investigator.agent.prompt import SYSTEM_PROMPT, build_decision_prompt
from ci_failure_investigator.models import Hypothesis, HypothesisStatus, InvestigationState


class DecisionPolicyError(RuntimeError):
    """Raised when an OpenAI response does not contain a structured decision."""


class _OpenAIHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    status: HypothesisStatus | None
    supporting_evidence_ids: list[str] | None
    contradicting_evidence_ids: list[str] | None
    revision_reason: str | None


class _OpenAIInvestigationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal["LIST", "SEARCH", "READ", "CONCLUDE"]
    path: str | None
    query: str | None
    file_glob: str | None
    start_line: int | None
    end_line: int | None
    hypotheses: list[_OpenAIHypothesis] | None


class _ParsedResponse(Protocol):
    output_parsed: _OpenAIInvestigationDecision | None


class _Responses(Protocol):
    def parse(
        self,
        *,
        model: str,
        input: str,
        instructions: str,
        text_format: type[_OpenAIInvestigationDecision],
    ) -> _ParsedResponse: ...


class _OpenAIClient(Protocol):
    @property
    def responses(self) -> _Responses: ...


class OpenAIDecisionPolicy:
    def __init__(self, client: _OpenAIClient, model: str) -> None:
        self._client = client
        self._model = model

    def __call__(self, state: InvestigationState) -> InvestigationDecision:
        response = self._client.responses.parse(
            model=self._model,
            input=build_decision_prompt(state),
            instructions=SYSTEM_PROMPT,
            text_format=_OpenAIInvestigationDecision,
        )
        transport_decision = response.output_parsed
        if transport_decision is None:
            raise DecisionPolicyError("OpenAI response did not contain a parsed InvestigationDecision")
        return _to_domain_decision(transport_decision)


def _to_domain_decision(
    transport_decision: _OpenAIInvestigationDecision,
) -> InvestigationDecision:
    try:
        if transport_decision.action_type == "LIST":
            action: Action = ListAction(path=transport_decision.path or ".")
        elif transport_decision.action_type == "SEARCH":
            if not transport_decision.query:
                raise DecisionPolicyError("SEARCH decision requires a query")
            action = SearchAction(
                query=transport_decision.query,
                file_glob=transport_decision.file_glob,
            )
        elif transport_decision.action_type == "READ":
            if (
                transport_decision.path is None
                or transport_decision.start_line is None
                or transport_decision.end_line is None
            ):
                raise DecisionPolicyError(
                    "READ decision requires path, start_line, and end_line"
                )
            action = ReadAction(
                path=transport_decision.path,
                start_line=transport_decision.start_line,
                end_line=transport_decision.end_line,
            )
        else:
            action = ConcludeAction()
        hypotheses = (
            None
            if transport_decision.hypotheses is None
            else [_to_domain_hypothesis(hypothesis) for hypothesis in transport_decision.hypotheses]
        )
        return InvestigationDecision(action=action, hypotheses=hypotheses)
    except DecisionPolicyError:
        raise
    except ValidationError as exc:
        raise DecisionPolicyError(
            f"Invalid {transport_decision.action_type} decision: {exc}"
        ) from exc


def _to_domain_hypothesis(transport_hypothesis: _OpenAIHypothesis) -> Hypothesis:
    try:
        return Hypothesis(
            id=transport_hypothesis.id,
            description=transport_hypothesis.description,
            status=transport_hypothesis.status or HypothesisStatus.UNVERIFIED,
            supporting_evidence_ids=transport_hypothesis.supporting_evidence_ids or [],
            contradicting_evidence_ids=transport_hypothesis.contradicting_evidence_ids or [],
            revision_reason=transport_hypothesis.revision_reason,
        )
    except ValidationError as exc:
        raise DecisionPolicyError(f"Invalid hypothesis in OpenAI decision: {exc}") from exc

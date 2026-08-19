from typing import Protocol

from ci_failure_investigator.agent.actions import InvestigationDecision
from ci_failure_investigator.agent.prompt import SYSTEM_PROMPT, build_decision_prompt
from ci_failure_investigator.models import InvestigationState


class DecisionPolicyError(RuntimeError):
    """Raised when an OpenAI response does not contain a structured decision."""


class _ParsedResponse(Protocol):
    output_parsed: InvestigationDecision | None


class _Responses(Protocol):
    def parse(
        self,
        *,
        model: str,
        input: str,
        instructions: str,
        text_format: type[InvestigationDecision],
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
            text_format=InvestigationDecision,
        )
        decision = response.output_parsed
        if decision is None:
            raise DecisionPolicyError("OpenAI response did not contain a parsed InvestigationDecision")
        return decision

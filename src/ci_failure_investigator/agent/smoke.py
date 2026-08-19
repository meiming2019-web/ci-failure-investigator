import os
from pathlib import Path
from typing import Any

from ci_failure_investigator.agent.graph import InvestigationRunResult, run_investigation
from ci_failure_investigator.agent.openai_policy import OpenAIDecisionPolicy
from ci_failure_investigator.logs import parse_ci_failure
from ci_failure_investigator.models import InvestigationState


def run_smoke_investigation(
    *,
    ci_log_text: str,
    repo_root: str | os.PathLike[str],
    client: Any,
    model: str,
    tool_call_budget: int = 8,
) -> InvestigationRunResult:
    failure = parse_ci_failure(ci_log_text)
    initial_state = InvestigationState(
        failure=failure,
        tool_call_budget=tool_call_budget,
    )
    policy = OpenAIDecisionPolicy(client, model=model)
    return run_investigation(initial_state, Path(repo_root), policy)

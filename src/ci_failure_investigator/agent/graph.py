from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from ci_failure_investigator.agent.actions import (
    Action,
    ConcludeAction,
    InvestigationDecision,
    ListAction,
    ReadAction,
    SearchAction,
)
from ci_failure_investigator.agent.trace import InvestigationTraceStep
from ci_failure_investigator.models import (
    Evidence,
    InvestigationState,
    RepositoryListResult,
    RepositoryReadResult,
    RepositorySearchResult,
)
from ci_failure_investigator.tools import (
    list_repository_path,
    read_repository_file,
    search_repository,
)


class TerminationReason(str, Enum):
    CONCLUDED = "CONCLUDED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class InvestigationRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: InvestigationState
    termination_reason: TerminationReason
    trace: list[InvestigationTraceStep] = Field(default_factory=list)


DecisionPolicy = Callable[[InvestigationState], InvestigationDecision]


class _GraphState(TypedDict):
    investigation: InvestigationState
    pending_action: Action | None
    termination_reason: TerminationReason | None
    trace: list[InvestigationTraceStep]


def _rebuild_state(state: InvestigationState, **updates: object) -> InvestigationState:
    values = state.model_dump()
    values.update(updates)
    return InvestigationState.model_validate(values)


def _next_evidence_id(state: InvestigationState) -> str:
    existing_ids = {evidence.id for evidence in state.evidence}
    number = 1
    while f"E{number}" in existing_ids:
        number += 1
    return f"E{number}"


def _format_list(result: RepositoryListResult) -> str:
    entries = result.entries
    if not entries:
        return "(empty)"
    return "\n".join(f"{entry.path} [{entry.entry_type.value}]" for entry in entries)


def _format_search(result: RepositorySearchResult) -> str:
    matches = result.matches
    observation = "(no matches)" if not matches else "\n".join(
        f"{match.path}:{match.line_number}: {match.line_text}" for match in matches
    )
    if result.truncated:
        observation += "\n(results truncated)"
    return observation


def _format_read(result: RepositoryReadResult) -> str:
    return result.content


def _decision_node(policy: DecisionPolicy, graph_state: _GraphState) -> dict[str, object]:
    current = graph_state["investigation"]
    if current.tool_calls_used >= current.tool_call_budget:
        return {
            "pending_action": None,
            "termination_reason": TerminationReason.BUDGET_EXHAUSTED,
        }

    decision = policy(_rebuild_state(current))
    trace_step = InvestigationTraceStep(
        iteration=len(graph_state["trace"]) + 1,
        decision=decision,
    )
    trace = [*graph_state["trace"], trace_step]
    updated = current
    if decision.hypotheses is not None:
        updated = _rebuild_state(current, hypotheses=decision.hypotheses)

    if isinstance(decision.action, ConcludeAction):
        return {
            "investigation": updated,
            "pending_action": None,
            "termination_reason": TerminationReason.CONCLUDED,
            "trace": trace,
        }
    return {"investigation": updated, "pending_action": decision.action, "trace": trace}


def _execute_node(repo_root: str | Path, graph_state: _GraphState) -> dict[str, object]:
    current = graph_state["investigation"]
    action = graph_state["pending_action"]
    if action is None:
        raise RuntimeError("execute node requires a pending action")

    if isinstance(action, ListAction):
        list_result = list_repository_path(repo_root, action.path)
        source_type = "repository_list"
        source_location = list_result.path
        observation = _format_list(list_result)
        metadata = {"operation": "LIST", "truncated": list_result.truncated}
    elif isinstance(action, SearchAction):
        search_result = search_repository(repo_root, action.query, action.file_glob)
        source_type = "repository_search"
        source_location = f"search:{action.query}"
        if action.file_glob is not None:
            source_location += f" [{action.file_glob}]"
        observation = _format_search(search_result)
        metadata = {
            "operation": "SEARCH",
            "query": action.query,
            "truncated": search_result.truncated,
        }
        if action.file_glob is not None:
            metadata["file_glob"] = action.file_glob
    elif isinstance(action, ReadAction):
        read_result = read_repository_file(repo_root, action.path, action.start_line, action.end_line)
        source_type = "repository_read"
        source_location = f"{read_result.path}:{read_result.start_line}-{read_result.end_line}"
        observation = _format_read(read_result)
        metadata = {"operation": "READ"}
    else:
        raise TypeError(f"unsupported repository action: {action.action_type}")

    new_tool_calls_used = current.tool_calls_used + 1
    evidence = Evidence(
        id=_next_evidence_id(current),
        source_type=source_type,
        source_location=source_location,
        observation=observation,
        step_number=new_tool_calls_used,
        metadata=metadata,
    )
    trace = [
        *graph_state["trace"][:-1],
        InvestigationTraceStep(
            iteration=graph_state["trace"][-1].iteration,
            decision=graph_state["trace"][-1].decision,
            evidence_id=evidence.id,
        ),
    ]
    updated = _rebuild_state(
        current,
        evidence=[*current.evidence, evidence],
        tool_calls_used=new_tool_calls_used,
    )
    return {"investigation": updated, "pending_action": None, "trace": trace}


def _route_after_decision(graph_state: _GraphState) -> str:
    if graph_state["termination_reason"] is not None:
        return END
    return "execute"


def _route_after_execution(graph_state: _GraphState) -> str:
    return "decide"


def run_investigation(
    initial_state: InvestigationState,
    repo_root: str | Path,
    decision_policy: DecisionPolicy,
) -> InvestigationRunResult:
    workflow = StateGraph(_GraphState)
    workflow.add_node("decide", lambda state: _decision_node(decision_policy, state))
    workflow.add_node("execute", lambda state: _execute_node(repo_root, state))
    workflow.add_edge(START, "decide")
    workflow.add_conditional_edges("decide", _route_after_decision)
    workflow.add_conditional_edges("execute", _route_after_execution)
    graph = workflow.compile()

    final_state = graph.invoke(
        {
            "investigation": initial_state,
            "pending_action": None,
            "termination_reason": None,
            "trace": [],
        },
        config={"recursion_limit": max(10, initial_state.tool_call_budget * 3 + 3)},
    )
    return InvestigationRunResult(
        state=final_state["investigation"],
        termination_reason=final_state["termination_reason"],
        trace=final_state["trace"],
    )

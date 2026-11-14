"""Steps 4-5: the agent loop. The LLM decides which retrieval tools to call; code enforces the
rules around it: evidence gate, two separate caps, retry limits, forced lookups, grounding check."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import ValidationError

from . import guard
from .config import Settings
from .evidence import Evidence
from .llm.base import Message, ToolCall, ToolChoice, ToolResponse, ToolSpec
from .llmcaller import LLMCaller, LLMUnavailable
from .prompts import AGENT_PROMPT, load_prompt
from .schemas import Classification, Feedback, ReportDraft
from .tools import (
    RETRIEVAL_TOOLS,
    SOURCES,
    ToolContext,
    ToolResult,
    ToolStatus,
    run_tool,
    tool_specs,
)
from .trace import Event, Tracer
from .validator import validate_draft

SUBMIT_TOOL = ToolSpec(
    "submit_report",
    "Submit the final report for the CS officer. Allowed only after guidelines, customer and policies "
    "were all consulted. The system validates every citation and action against the tool results; if "
    "it is rejected, fix the listed problems and call it again.",
    ReportDraft.model_json_schema(),
)

_PERSISTENT = ("source_unavailable", "source_failure")  # a source problem, not an LLM mistake
_QSTOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "i",
    "we",
    "you",
    "my",
    "our",
    "is",
    "are",
    "was",
    "it",
    "in",
    "on",
    "for",
    "with",
    "this",
    "that",
    "have",
    "has",
    "not",
    "your",
    "me",
    "be",
    "at",
    "as",
}


def policy_query(classification: Classification, feedback: Feedback) -> str:
    """Deterministic keyword query used when the system (not the LLM) has to search policies.

    Example: billing_issue + "I was charged twice for invoice INV-5005, please refund!"
    -> "billing issue charged twice invoice inv 5005 please refund".
    """
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", feedback.text.lower())
        if word not in _QSTOP and len(word) > 2
    ]
    return (
        classification.category.value.replace("_", " ") + " " + " ".join(dict.fromkeys(words))[:80]
    ).strip()


@dataclass
class RunState:
    feedback: Feedback
    classification: Classification
    tool_context: ToolContext
    evidence: Evidence
    tracer: Tracer
    flags: set[str] = field(default_factory=set)


class ToolExecutor:
    """Runs tool calls with caching, budgets, one retry for transient errors and tracing."""

    def __init__(self, run_state: RunState, max_tool_calls: int, max_forced_lookups: int):
        self.run_state = run_state
        self.max_tool_calls, self.max_forced_lookups = max_tool_calls, max_forced_lookups
        self.llm_tool_calls = self.forced_calls = self.cache_hits = self.degraded_calls = 0
        self.failed_sources: dict[str, str] = (
            {}
        )  # source -> error, after the single retry also failed
        self._cache: dict[str, ToolResult] = {}

    @staticmethod
    def _cache_key(name: str, args: dict) -> str:
        """Tool name + args with strings trimmed and lowercased, keys sorted.

        So {"email": " A@x.com"} and {"email": "a@x.com"} share one cache entry.
        """
        normalized_args = {
            k: (v.strip().lower() if isinstance(v, str) else v) for k, v in (args or {}).items()
        }
        return name + json.dumps(normalized_args, sort_keys=True, default=str)

    def call(self, name: str, args: dict[str, Any], origin: str = "llm") -> ToolResult:
        """Run one tool call under the run's rules and return its result (never raises).

        Order: cache hit -> budget (only origin="llm" counts against max_tool_calls) -> run_tool,
        retried once if retryable -> _finish. origin: llm, forced_by_system (last-turn lookups)
        or degraded (no-LLM path).
        """
        tracer, cache_key = self.run_state.tracer, self._cache_key(name, args)
        if cache_key in self._cache:  # same call twice in a run: no new work, no budget
            self.cache_hits += 1
            tracer.emit(Event.tool_call, tool=name, args=args, origin=origin, cache_hit=True)
            return self._cache[cache_key]
        if origin == "llm" and self.llm_tool_calls >= self.max_tool_calls:
            tracer.emit(
                Event.tool_error, tool=name, args=args, origin=origin, error="tool_budget_exhausted"
            )
            return ToolResult(
                name,
                SOURCES.get(name, "unknown"),
                ToolStatus.error,
                error="tool_budget_exhausted",
                message=f"tool call budget ({self.max_tool_calls}) used up; call submit_report",
            )
        # unreachable today: there are only 3 sources to force
        if (
            origin == "forced_by_system" and self.forced_calls >= self.max_forced_lookups
        ):  # pragma: no cover
            return ToolResult(
                name,
                SOURCES.get(name, "unknown"),
                ToolStatus.error,
                error="forced_budget_exhausted",
            )
        if origin == "llm":
            self.llm_tool_calls += 1
        elif origin == "forced_by_system":
            self.forced_calls += 1
        else:
            self.degraded_calls += 1
        for attempt in (1, 2):
            tracer.emit(
                Event.tool_call,
                tool=name,
                args=args,
                origin=origin,
                attempt=attempt,
                cache_hit=False,
            )
            tool_result = run_tool(self.run_state.tool_context, name, args)
            if tool_result.status is ToolStatus.error:
                tracer.emit(
                    Event.tool_error,
                    tool=name,
                    error=tool_result.error,
                    message=tool_result.message,
                    attempt=attempt,
                    will_retry=tool_result.retryable and attempt == 1,
                )
                if tool_result.retryable and attempt == 1:
                    continue  # one retry of a transient source error; it costs no extra budget
            else:
                tracer.emit(
                    Event.tool_result,
                    tool=name,
                    status=tool_result.status.value,
                    data=tool_result.data,
                    message=tool_result.message,
                )
            break
        self._finish(tool_result, name, cache_key)
        return tool_result

    def _finish(self, tool_result: ToolResult, name: str, cache_key: str) -> None:
        """Book a result: evidence, flags, failed sources and an injection scan of its data.

        Successes and LLM mistakes (out_of_scope, invalid_arguments) are cached; source errors are
        not, so a later call can still succeed.
        """
        self.run_state.evidence.record(tool_result)
        self.run_state.flags.update(tool_result.flags)
        if tool_result.status is ToolStatus.error:
            if tool_result.error in _PERSISTENT:
                self.failed_sources[tool_result.source] = tool_result.error
        else:
            self.failed_sources.pop(tool_result.source, None)
            hits = guard.scan(json.dumps(tool_result.data, default=str)) if tool_result.data else []
            if hits:
                self.run_state.flags.add("injection_in_tool_result")
                self.run_state.tracer.emit(
                    Event.injection_guard, source=f"tool:{name}", patterns=hits
                )
        if tool_result.status is not ToolStatus.error or tool_result.error in (
            "out_of_scope",
            "invalid_arguments",
        ):
            self._cache[cache_key] = (
                tool_result  # transient/source errors are not cached, so a later call can succeed
            )

    def counters(self) -> dict[str, int]:
        """Tool counters for the report (tool_calls = calls the LLM made)."""
        return {
            "tool_calls": self.llm_tool_calls,
            "forced_by_system": self.forced_calls,
            "cache_hits": self.cache_hits,
            "degraded_calls": self.degraded_calls,
        }


@dataclass
class AgentResult:
    outcome: str  # submitted | grounding_failed | budget_exhausted | llm_unavailable | incomplete_context
    draft: Optional[ReportDraft] = None
    rejected_draft: Optional[dict] = None
    llm_turns: int = 0
    failed_sources: dict[str, str] = field(default_factory=dict)


class AgentLoop:
    """The multi-turn conversation in which the LLM gathers evidence and submits a report."""

    def __init__(
        self,
        run_state: RunState,
        tool_executor: ToolExecutor,
        llm_caller: LLMCaller,
        settings: Settings,
    ):
        self.run_state, self.settings = run_state, settings
        self.tool_executor, self.llm_caller = tool_executor, llm_caller

    # -------------------------------------------------------------- prompts
    def _opening(self) -> str:
        """First user message: escaped feedback, metadata, fixed classification, flags, budget."""
        feedback, classification = self.run_state.feedback, self.run_state.classification
        alternatives = (
            ", ".join(
                f"{alternative.category.value} ({alternative.confidence:.2f})"
                for alternative in classification.alternatives
            )
            or "none"
        )
        flags = sorted(self.run_state.flags) or ["none"]
        return (
            f"{guard.wrap_feedback(feedback.text)}\n\n"
            f"Metadata: customer_email={guard.escape(feedback.customer_email)}; channel={feedback.channel.value}; "
            f"received_at={feedback.received_at.isoformat()}; feedback_id={feedback.feedback_id}\n"
            f"Classification (set by an earlier step and by the system): category={classification.category.value}; "
            f"alternatives={alternatives}; sentiment={classification.sentiment.value}; "
            f"urgency={classification.urgency.value}; "
            f"confidence={classification.confidence:.2f}; needs_human_triage={classification.needs_human_triage}\n"
            f"Flags: {', '.join(flags)}\n"
            f"Budget: at most {self.settings.max_llm_turns} turns (the last one is reserved for submit_report) and "
            f"{self.settings.max_tool_calls} tool calls. Consult the three sources, then submit the report."
        )

    @staticmethod
    def _tool_call_response_body(tool_result: ToolResult, origin: str = "llm") -> dict[str, Any]:
        """The reply the LLM receives for one tool call (sent as Gemini's functionResponse).

        The result goes in escaped <tool_result> tags so the LLM reads it as data, not orders;
        status (and error, if any) are repeated at the top level. Example, refused lookup:
        {"status": "error", "error": "out_of_scope", "content": "<tool_result>...</tool_result>"}
        """
        payload: dict[str, Any] = {
            "status": tool_result.status.value,
            "content": guard.wrap_tool_result(tool_result.tool, tool_result.to_dict(), origin),
        }
        if tool_result.error:
            payload["error"] = tool_result.error
        return payload

    def _forced_lookups(self, messages: list[Message]) -> None:
        """Last turn with sources still missing: the system runs those lookups itself.

        Arguments come from code (category, metadata email, policy_query); the results are added
        to the last message and step_limit_reached is flagged. No extra LLM turn is used.
        """
        run_state, evidence = self.run_state, self.run_state.evidence
        plan = {
            "guidelines": (
                "get_cs_guidelines",
                {"category": run_state.classification.category.value},
            ),
            "customer": ("lookup_customer", {"email": run_state.feedback.customer_email}),
            "policies": (
                "search_policies",
                {
                    "query": policy_query(run_state.classification, run_state.feedback),
                    "category": run_state.classification.category.value,
                },
            ),
        }
        blocks = []
        run_state.flags.add(
            "step_limit_reached"
        )  # the cap forced the system to do the retrieval itself
        for source in list(evidence.missing_sources):
            name, args = plan[source]
            tool_result = self.tool_executor.call(name, args, origin="forced_by_system")
            blocks.append(guard.wrap_tool_result(name, tool_result.to_dict(), "forced_by_system"))
        note = (
            "The system ran the lookups you had not done yet, because the last turn is reserved for "
            "submit_report. Their results:\n" + "\n".join(blocks) + "\nNow call submit_report."
        )
        last = messages[-1]
        last.text = (last.text + "\n\n" if last.text else "") + note

    def loop(self) -> AgentResult:
        """Converse until a draft is accepted or a stop rule fires; return the outcome.

        Outcomes: submitted, grounding_failed, incomplete_context, llm_unavailable,
        budget_exhausted. The last turn only offers submit_report (ToolChoice "required").
        """
        run_state, evidence, settings = self.run_state, self.run_state.evidence, self.settings
        messages = [Message("system", load_prompt(AGENT_PROMPT)), Message("user", self._opening())]
        fix_used, turns = False, 0
        for turn in range(1, settings.max_llm_turns + 1):
            final_turn = turn == settings.max_llm_turns
            if (
                self.tool_executor.failed_sources
            ):  # a source stayed broken after its retry: report incomplete_context
                return AgentResult(
                    "incomplete_context",
                    llm_turns=turns,
                    failed_sources=dict(self.tool_executor.failed_sources),
                )

            if final_turn:
                if evidence.missing_sources:
                    self._forced_lookups(messages)
                    if evidence.missing_sources:
                        failed_sources = dict(self.tool_executor.failed_sources) or dict(
                            evidence.source_errors
                        )
                        return AgentResult(
                            "incomplete_context", llm_turns=turns, failed_sources=failed_sources
                        )
                tools, tool_choice = [SUBMIT_TOOL], ToolChoice("required", ["submit_report"])
            else:
                tools, tool_choice = tool_specs(SUBMIT_TOOL), ToolChoice("auto")

            try:
                llm_turn = self.llm_caller.call(
                    "agent_turn", AGENT_PROMPT, messages, tools, tool_choice, turn
                )
            except LLMUnavailable:
                return AgentResult("llm_unavailable", llm_turns=turns)

            if any(
                not isinstance(call.args, dict) for call in llm_turn.tool_calls
            ):  # provider returned an unusable call
                run_state.tracer.emit(
                    Event.llm_error,
                    purpose="agent_turn",
                    turn=turn,
                    error="tool call args are not an object",
                )
                return AgentResult("llm_unavailable", llm_turns=turns)

            turns = turn
            messages.append(llm_turn.as_message())
            if not llm_turn.tool_calls:
                messages.append(
                    Message(
                        "user",
                        "Use the tools to consult guidelines, customer and policies, "
                        "then call submit_report.",
                    )
                )
                continue
            responses, draft, verdict = self._reply_to_tool_calls(
                llm_turn.tool_calls, final_turn, fix_used
            )
            messages.append(Message("tool", tool_responses=responses))
            if verdict == "fix":
                fix_used = True
            if draft is not None:
                return AgentResult("submitted", draft=draft, llm_turns=turns)
            if verdict == "grounding_failed":
                return AgentResult(
                    "grounding_failed", rejected_draft=self._rejected_draft, llm_turns=turns
                )
        return AgentResult("budget_exhausted", llm_turns=turns)

    _rejected_draft: Optional[dict] = None

    def _reply_to_tool_calls(self, calls: list[ToolCall], final_turn: bool, fix_used: bool):
        """Reply to each tool call of a model turn -> (responses in call order, draft, verdict).

        Retrieval calls run first. submit_report in the same turn as retrieval -> submit_too_early,
        a second submit -> duplicate_submit. verdict: None, "gate", "fix" or "grounding_failed".
        """
        response_by_call_id: dict[str, dict] = {}
        retrieval_calls = [call for call in calls if call.name != "submit_report"]
        submit_calls = [call for call in calls if call.name == "submit_report"]
        for call in retrieval_calls:
            response_by_call_id[call.id] = self._tool_call_response_body(
                self.tool_executor.call(call.name, call.args)
            )
        draft, verdict = None, None
        for i, call in enumerate(submit_calls):
            if retrieval_calls:
                response_by_call_id[call.id] = {
                    "status": "error",
                    "error": "submit_too_early",
                    "message": "submit_report was called in the same turn as retrieval tools; "
                    "call it again after reading their results.",
                }
            elif i > 0:
                response_by_call_id[call.id] = {
                    "status": "error",
                    "error": "duplicate_submit",
                    "message": "only one submit_report per turn.",
                }
            else:
                response_by_call_id[call.id], draft, verdict = self._check_submit_report(
                    call, final_turn, fix_used
                )
        return (
            [ToolResponse(call.id, call.name, response_by_call_id[call.id]) for call in calls],
            draft,
            verdict,
        )

    def _check_submit_report(self, call: ToolCall, final_turn: bool, fix_used: bool):
        """Check one submit_report: evidence gate, schema, then validate_draft.

        Returns (response, draft if accepted, verdict). One fix is allowed: the first failure sends
        the issues back ("fix"); a second failure, or any failure on the last turn, ends the run
        ("grounding_failed"). A gate refusal does not use up the fix.
        """
        evidence, tracer = self.run_state.evidence, self.run_state.tracer
        if evidence.missing_sources:  # evidence gate: not_found counts as consulted, error does not
            tracer.emit(
                Event.grounding_validation,
                ok=False,
                gate="missing_sources",
                missing=evidence.missing_sources,
            )
            return (
                {
                    "status": "error",
                    "error": "missing_sources",
                    "missing_sources": evidence.missing_sources,
                    "message": "Consult these sources first: guidelines=get_cs_guidelines, "
                    "customer=lookup_customer, policies=search_policies.",
                },
                None,
                "gate",
            )
        try:
            draft = ReportDraft.model_validate(call.args)
            issues = validate_draft(draft, evidence, self.run_state.feedback.text)
        except ValidationError as e:
            draft, issues = (
                None,
                [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()][:8],
            )
        attempt = 2 if fix_used else 1
        no_repair_left = bool(issues) and (fix_used or final_turn)
        tracer.emit(
            Event.grounding_validation,
            ok=not issues,
            attempt=attempt,
            issues=issues,
            **({"final": True, "rejected_draft": call.args} if no_repair_left else {}),
        )
        if not issues:
            return {"status": "accepted"}, draft, None
        if (
            no_repair_left
        ):  # caller rebuilds the report from validated data; the bad draft stays in the trace
            self._rejected_draft = call.args
            return (
                {"status": "error", "error": "validation_failed", "issues": issues},
                None,
                "grounding_failed",
            )
        return (
            {
                "status": "error",
                "error": "validation_failed",
                "issues": issues,
                "message": "Fix exactly these problems and call submit_report again.",
            },
            None,
            "fix",
        )

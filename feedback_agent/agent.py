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
from .tools import RETRIEVAL_TOOLS, SOURCES, ToolContext, ToolResult, ToolStatus, run_tool, tool_specs
from .trace import Event, Tracer
from .validator import validate_draft

SUBMIT_TOOL = ToolSpec(
    "submit_report",
    "Submit the final report for the CS officer. Allowed only after guidelines, customer and policies "
    "were all consulted. The system validates every citation and action against the tool results; if "
    "it is rejected, fix the listed problems and call it again.",
    ReportDraft.model_json_schema())

_PERSISTENT = ("source_unavailable", "source_failure")  # a source problem, not an LLM mistake
_QSTOP = {"the", "a", "an", "and", "or", "to", "of", "i", "we", "you", "my", "our", "is", "are", "was", "it",
          "in", "on", "for", "with", "this", "that", "have", "has", "not", "your", "me", "be", "at", "as"}


def policy_query(cls: Classification, fb: Feedback) -> str:
    """Deterministic keyword query used when the system (not the LLM) has to search policies."""
    words = [w for w in re.findall(r"[a-z0-9]+", fb.text.lower()) if w not in _QSTOP and len(w) > 2]
    return (cls.category.value.replace("_", " ") + " " + " ".join(dict.fromkeys(words))[:80]).strip()


@dataclass
class RunState:
    fb: Feedback
    cls: Classification
    ctx: ToolContext
    ev: Evidence
    tracer: Tracer
    flags: set[str] = field(default_factory=set)


class ToolExecutor:
    """Runs tool calls with caching, budgets, one retry for transient errors and tracing."""

    def __init__(self, st: RunState, max_tool_calls: int, max_forced: int):
        self.st, self.max_tool_calls, self.max_forced = st, max_tool_calls, max_forced
        self.used = self.forced = self.cache_hits = self.degraded = 0
        self.failed: dict[str, str] = {}  # source -> error, after the single retry also failed
        self._cache: dict[str, ToolResult] = {}

    @staticmethod
    def _key(name: str, args: dict) -> str:
        norm = {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in (args or {}).items()}
        return name + json.dumps(norm, sort_keys=True, default=str)

    def call(self, name: str, args: dict[str, Any], origin: str = "llm") -> ToolResult:
        tr, key = self.st.tracer, self._key(name, args)
        if key in self._cache:  # same call twice in a run: no new work, no budget
            self.cache_hits += 1
            tr.emit(Event.tool_call, tool=name, args=args, origin=origin, cache_hit=True)
            return self._cache[key]
        if origin == "llm" and self.used >= self.max_tool_calls:
            tr.emit(Event.tool_error, tool=name, args=args, origin=origin, error="tool_budget_exhausted")
            return ToolResult(name, SOURCES.get(name, "unknown"), ToolStatus.error, error="tool_budget_exhausted",
                              message=f"tool call budget ({self.max_tool_calls}) used up; call submit_report")
        if origin == "forced_by_system" and self.forced >= self.max_forced:  # pragma: no cover - 3 sources
            return ToolResult(name, SOURCES.get(name, "unknown"), ToolStatus.error, error="forced_budget_exhausted")
        if origin == "llm":
            self.used += 1
        elif origin == "forced_by_system":
            self.forced += 1
        else:
            self.degraded += 1
        for attempt in (1, 2):
            tr.emit(Event.tool_call, tool=name, args=args, origin=origin, attempt=attempt, cache_hit=False)
            r = run_tool(self.st.ctx, name, args)
            if r.status is ToolStatus.error:
                tr.emit(Event.tool_error, tool=name, error=r.error, message=r.message, attempt=attempt,
                        will_retry=r.retryable and attempt == 1)
                if r.retryable and attempt == 1:
                    continue  # one retry of a transient source error; it costs no extra budget
            else:
                tr.emit(Event.tool_result, tool=name, status=r.status.value, data=r.data, message=r.message)
            break
        self._finish(r, name, key)
        return r

    def _finish(self, r: ToolResult, name: str, key: str) -> None:
        self.st.ev.record(r)
        self.st.flags.update(r.flags)
        if r.status is ToolStatus.error:
            if r.error in _PERSISTENT:
                self.failed[r.source] = r.error
        else:
            self.failed.pop(r.source, None)
            hits = guard.scan(json.dumps(r.data, default=str)) if r.data else []
            if hits:
                self.st.flags.add("injection_in_tool_result")
                self.st.tracer.emit(Event.injection_guard, source=f"tool:{name}", patterns=hits)
        if r.status is not ToolStatus.error or r.error in ("out_of_scope", "invalid_arguments"):
            self._cache[key] = r  # transient/source errors are not cached, so a later call can succeed

    def counters(self) -> dict[str, int]:
        return {"tool_calls": self.used, "forced_by_system": self.forced, "cache_hits": self.cache_hits,
                "degraded_calls": self.degraded}


@dataclass
class AgentResult:
    outcome: str  # submitted | grounding_failed | budget_exhausted | llm_unavailable | incomplete_context
    draft: Optional[ReportDraft] = None
    rejected_draft: Optional[dict] = None
    llm_turns: int = 0
    failed_sources: dict[str, str] = field(default_factory=dict)


class AgentLoop:
    def __init__(self, st: RunState, executor: ToolExecutor, caller: LLMCaller, s: Settings):
        self.st, self.ex, self.caller, self.s = st, executor, caller, s

    # -------------------------------------------------------------- prompts
    def _opening(self) -> str:
        fb, c = self.st.fb, self.st.cls
        alts = ", ".join(f"{a.category.value} ({a.confidence:.2f})" for a in c.alternatives) or "none"
        flags = sorted(self.st.flags) or ["none"]
        return (
            f"{guard.wrap_feedback(fb.text)}\n\n"
            f"Metadata: customer_email={guard.escape(fb.customer_email)}; channel={fb.channel.value}; "
            f"received_at={fb.received_at.isoformat()}; feedback_id={fb.feedback_id}\n"
            f"Classification (set by an earlier step and by the system): category={c.category.value}; "
            f"alternatives={alts}; sentiment={c.sentiment.value}; urgency={c.urgency.value}; "
            f"confidence={c.confidence:.2f}; needs_human_triage={c.needs_human_triage}\n"
            f"Flags: {', '.join(flags)}\n"
            f"Budget: at most {self.s.max_llm_turns} turns (the last one is reserved for submit_report) and "
            f"{self.s.max_tool_calls} tool calls. Consult the three sources, then submit the report.")

    @staticmethod
    def _payload(r: ToolResult, origin: str = "llm") -> dict[str, Any]:
        out: dict[str, Any] = {"status": r.status.value,
                               "content": guard.wrap_tool_result(r.tool, r.to_dict(), origin)}
        if r.error:
            out["error"] = r.error
        return out

    def _forced_lookups(self, messages: list[Message]) -> None:
        st, ev = self.st, self.st.ev
        plan = {"guidelines": ("get_cs_guidelines", {"category": st.cls.category.value}),
                "customer": ("lookup_customer", {"email": st.fb.customer_email}),
                "policies": ("search_policies", {"query": policy_query(st.cls, st.fb),
                                                 "category": st.cls.category.value})}
        blocks = []
        st.flags.add("step_limit_reached")  # the cap forced the system to do the retrieval itself
        for src in list(ev.missing_sources):
            name, args = plan[src]
            r = self.ex.call(name, args, origin="forced_by_system")
            blocks.append(guard.wrap_tool_result(name, r.to_dict(), "forced_by_system"))
        note = ("The system ran the lookups you had not done yet, because the last turn is reserved for "
                "submit_report. Their results:\n" + "\n".join(blocks) + "\nNow call submit_report.")
        last = messages[-1]
        last.text = (last.text + "\n\n" if last.text else "") + note

    # ----------------------------------------------------------------- loop
    def run(self) -> AgentResult:
        st, ev, s = self.st, self.st.ev, self.s
        messages = [Message("system", load_prompt(AGENT_PROMPT)), Message("user", self._opening())]
        fix_used, turns = False, 0
        for turn in range(1, s.max_llm_turns + 1):
            final = turn == s.max_llm_turns
            if self.ex.failed:  # a source stayed broken after its retry: stop, report incomplete_context
                return AgentResult("incomplete_context", llm_turns=turns, failed_sources=dict(self.ex.failed))
            if final:
                if ev.missing_sources:
                    self._forced_lookups(messages)
                    if ev.missing_sources:
                        return AgentResult("incomplete_context", llm_turns=turns,
                                           failed_sources=dict(self.ex.failed) or dict(ev.source_errors))
                tools, choice = [SUBMIT_TOOL], ToolChoice("any", ["submit_report"])
            else:
                tools, choice = tool_specs(SUBMIT_TOOL), ToolChoice("auto")
            try:
                out = self.caller.call("agent_turn", AGENT_PROMPT, messages, tools, choice, turn)
            except LLMUnavailable:
                return AgentResult("llm_unavailable", llm_turns=turns)
            if any(not isinstance(c.args, dict) for c in out.tool_calls):  # provider returned an unusable call
                st.tracer.emit(Event.llm_error, purpose="agent_turn", turn=turn, error="tool call args are not an object")
                return AgentResult("llm_unavailable", llm_turns=turns)
            turns = turn
            messages.append(out.as_message())
            if not out.tool_calls:
                messages.append(Message("user", "Use the tools to consult guidelines, customer and policies, "
                                                "then call submit_report."))
                continue
            responses, draft, verdict = self._answer(out.tool_calls, final, fix_used)
            messages.append(Message("tool", tool_responses=responses))
            if verdict == "fix":
                fix_used = True
            if draft is not None:
                return AgentResult("submitted", draft=draft, llm_turns=turns)
            if verdict == "grounding_failed":
                return AgentResult("grounding_failed", rejected_draft=self._rejected, llm_turns=turns)
        return AgentResult("budget_exhausted", llm_turns=turns)

    _rejected: Optional[dict] = None

    def _answer(self, calls: list[ToolCall], final: bool, fix_used: bool):
        by_id: dict[str, dict] = {}
        retrieval = [c for c in calls if c.name != "submit_report"]
        submits = [c for c in calls if c.name == "submit_report"]
        for c in retrieval:
            by_id[c.id] = self._payload(self.ex.call(c.name, c.args))
        draft, verdict = None, None
        for i, c in enumerate(submits):
            if retrieval:
                by_id[c.id] = {"status": "error", "error": "submit_too_early",
                               "message": "submit_report was called in the same turn as retrieval tools; "
                                          "call it again after reading their results."}
            elif i > 0:
                by_id[c.id] = {"status": "error", "error": "duplicate_submit",
                               "message": "only one submit_report per turn."}
            else:
                by_id[c.id], draft, verdict = self._handle_submit(c, final, fix_used)
        return [ToolResponse(c.id, c.name, by_id[c.id]) for c in calls], draft, verdict

    def _handle_submit(self, call: ToolCall, final: bool, fix_used: bool):
        ev, tr = self.st.ev, self.st.tracer
        if ev.missing_sources:  # evidence gate: not_found counts as consulted, error does not
            tr.emit(Event.grounding_validation, ok=False, gate="missing_sources", missing=ev.missing_sources)
            return ({"status": "error", "error": "missing_sources", "missing_sources": ev.missing_sources,
                     "message": "Consult these sources first: guidelines=get_cs_guidelines, "
                                "customer=lookup_customer, policies=search_policies."}, None, "gate")
        try:
            draft = ReportDraft.model_validate(call.args)
            issues = validate_draft(draft, ev, self.st.fb.text)
        except ValidationError as e:
            draft, issues = None, [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()][:8]
        attempt = 2 if fix_used else 1
        no_repair_left = bool(issues) and (fix_used or final)
        tr.emit(Event.grounding_validation, ok=not issues, attempt=attempt, issues=issues,
                **({"final": True, "rejected_draft": call.args} if no_repair_left else {}))
        if not issues:
            return {"status": "accepted"}, draft, None
        if no_repair_left:  # caller rebuilds the report from validated data; the bad draft stays in the trace
            self._rejected = call.args
            return {"status": "error", "error": "validation_failed", "issues": issues}, None, "grounding_failed"
        return ({"status": "error", "error": "validation_failed", "issues": issues,
                 "message": "Fix exactly these problems and call submit_report again."}, None, "fix")

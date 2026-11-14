"""Tool executor: runs every retrieval call of a run, whether the LLM, the system's forced
last-turn lookups or the degraded path asked for it, under one set of rules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..ops.trace import Event, Tracer
from ..schemas import Classification, Feedback
from .evidence import Evidence
from .injection_scan import find_injection_patterns
from .tools import SOURCES, ToolContext, ToolResult, ToolStatus, run_tool

_SOURCE_FAILURE_ERRORS = (
    "source_unavailable",
    "source_failure",
)  # a source problem, not an LLM mistake


@dataclass
class CaseFile:
    """Everything one feedback's processing reads and adds to: the feedback, its classification,
    the tool context, the evidence gathered, the trace and the flags."""

    feedback: Feedback
    classification: Classification
    tool_context: ToolContext
    evidence: Evidence
    tracer: Tracer
    flags: set[str] = field(default_factory=set)


class ToolExecutor:
    """Runs tool calls with caching, budgets, one retry for transient errors and tracing."""

    def __init__(self, case_file: CaseFile, max_tool_calls: int, max_forced_lookups: int):
        self.case_file = case_file
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
        retried once if retryable -> _record_result. origin: llm, forced_by_system (last-turn lookups)
        or degraded (no-LLM path).
        """
        tracer, cache_key = self.case_file.tracer, self._cache_key(name, args)
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
            tool_result = run_tool(self.case_file.tool_context, name, args)
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
        self._record_result(tool_result, name, cache_key)
        return tool_result

    def _record_result(self, tool_result: ToolResult, name: str, cache_key: str) -> None:
        """Book a result: evidence, flags, failed sources and an injection scan of its data.

        Successes and LLM mistakes (out_of_scope, invalid_arguments) are cached; source errors are
        not, so a later call can still succeed.
        """
        self.case_file.evidence.record(tool_result)
        self.case_file.flags.update(tool_result.flags)
        if tool_result.status is ToolStatus.error:
            if tool_result.error in _SOURCE_FAILURE_ERRORS:
                self.failed_sources[tool_result.source] = tool_result.error
        else:
            self.failed_sources.pop(tool_result.source, None)
            hits = (
                find_injection_patterns(
                    json.dumps(tool_result.data, ensure_ascii=False, default=str)
                )
                if tool_result.data
                else []
            )
            if hits:
                self.case_file.flags.add("injection_in_tool_result")
                self.case_file.tracer.emit(
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

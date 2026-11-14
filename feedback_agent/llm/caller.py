"""One place that calls the provider: retry once on transient errors, trace every attempt."""

from __future__ import annotations

import time
from typing import Optional

from ..config import Settings
from ..ops.trace import Event, Tracer
from .base import LLMError, LLMProvider, LLMTurn, Message, ToolMode, ToolSpec


class LLMUnavailable(RuntimeError):
    """The provider failed (after the single allowed retry); the caller must degrade."""


class LLMCaller:
    """The one place that calls the LLM provider for a run."""

    def __init__(self, provider: LLMProvider, settings: Settings, tracer: Tracer):
        self.provider, self.settings, self.tracer = provider, settings, tracer

    def call(
        self,
        purpose: str,
        prompt_version: str,
        messages: list[Message],
        tools: list[ToolSpec],
        tool_mode: ToolMode,
        turn: Optional[int] = None,
    ) -> LLMTurn:
        """Call the provider, retrying once on a transient error; every attempt is traced.

        `purpose` (classify / agent_turn) and `turn` only label the trace. Raises LLMUnavailable
        on a permanent error or when the retry fails too.
        """
        # A retry of the same request does not use up an agent turn (it has its own cap of 1).
        for attempt in (1, 2):
            started = time.monotonic()
            try:
                llm_turn = self.provider.generate(
                    messages, tools, tool_mode, self.settings.max_output_tokens
                )
            except LLMError as e:
                self.tracer.record_failed_attempt(
                    getattr(e, "usage", None),
                    getattr(e, "usage_unknown", False),
                    purpose=purpose,
                    turn=turn,
                    attempt=attempt,
                    provider=self.provider.name,
                    transient=e.transient,
                    error=str(e)[:300],
                )
                if e.transient and attempt == 1:
                    time.sleep(self.settings.llm_retry_backoff_s)
                    continue
                raise LLMUnavailable(str(e)) from e
            self.tracer.record_llm_call(
                llm_turn.usage,
                purpose=purpose,
                turn=turn,
                attempt=attempt,
                provider=self.provider.name,
                model=self.provider.model,
                prompt_version=prompt_version,
                latency_ms=int((time.monotonic() - started) * 1000),
                tool_calls=[tool_call.name for tool_call in llm_turn.tool_calls],
                text_chars=len(llm_turn.text),
            )
            return llm_turn
        raise LLMUnavailable("unreachable")  # pragma: no cover

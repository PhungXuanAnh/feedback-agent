"""One place that calls the provider: retry once on transient errors, trace every attempt."""
from __future__ import annotations

import time
from typing import Optional

from .config import Settings
from .llm.base import LLMError, LLMProvider, LLMTurn, Message, ToolChoice, ToolSpec
from .trace import Event, Tracer


class LLMUnavailable(RuntimeError):
    """The provider failed (after the single allowed retry); the caller must degrade."""


class LLMCaller:
    def __init__(self, provider: LLMProvider, settings: Settings, tracer: Tracer):
        self.provider, self.s, self.tracer = provider, settings, tracer

    def call(self, purpose: str, prompt_version: str, messages: list[Message], tools: list[ToolSpec],
             choice: ToolChoice, turn: Optional[int] = None) -> LLMTurn:
        # A retry of the same request does not use up an agent turn (it has its own cap of 1).
        for attempt in (1, 2):
            t0 = time.monotonic()
            try:
                out = self.provider.generate(messages, tools, choice, self.s.max_output_tokens)
            except LLMError as e:
                self.tracer.record_failed_attempt(
                    getattr(e, "usage", None), getattr(e, "usage_unknown", False), purpose=purpose, turn=turn,
                    attempt=attempt, provider=self.provider.name, transient=e.transient, error=str(e)[:300])
                if e.transient and attempt == 1:
                    time.sleep(self.s.llm_retry_backoff_s)
                    continue
                raise LLMUnavailable(str(e)) from e
            self.tracer.record_llm_call(
                out.usage, purpose=purpose, turn=turn, attempt=attempt, provider=self.provider.name,
                model=self.provider.model, prompt_version=prompt_version,
                latency_ms=int((time.monotonic() - t0) * 1000),
                tool_calls=[c.name for c in out.tool_calls], text_chars=len(out.text))
            return out
        raise LLMUnavailable("unreachable")  # pragma: no cover

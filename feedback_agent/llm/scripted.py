"""ScriptedProvider: deterministic provider for tests and offline demos.

It replays a fixed list of turns, but it is deliberately strict about the conversation it is
given (calls answered by responses in the same order, tool choice respected) so it cannot hide
a bug in how the agent loop talks to a real provider."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from .base import LLMError, LLMTurn, Message, ToolCall, ToolChoice, ToolSpec


class ScriptError(AssertionError):
    """The agent loop did something a real provider would reject (not an LLMError on purpose)."""


Step = Union[LLMTurn, dict, Exception]


class ScriptedProvider:
    name = "scripted"
    model = "scripted-v1"

    def __init__(self, script: list[Step]):
        self._script = list(script)
        self.calls = 0
        self.seen: list[list[Message]] = []

    @classmethod
    def from_file(cls, path: str | Path) -> "ScriptedProvider":
        return cls(json.loads(Path(path).read_text("utf-8")))

    @staticmethod
    def check_conversation(messages: list[Message]) -> None:
        body = [m for m in messages if m.role != "system"]
        if not body or body[0].role != "user":
            raise ScriptError("conversation must start with a user message")
        for prev, cur in zip(body, body[1:]):
            if prev.role == "model" and prev.tool_calls:
                want = [c.id for c in prev.tool_calls]
                got = [r.call_id for r in cur.tool_responses] if cur.role == "tool" else None
                if got != want:
                    raise ScriptError(f"function calls {want} must be answered by one tool message with the "
                                      f"same ids in order, got {got}")
            elif cur.role == "tool":
                raise ScriptError("tool message without a preceding model turn that made calls")
            elif cur.role == prev.role == "model":
                raise ScriptError("two consecutive model turns")
        if body[-1].role == "model":
            raise ScriptError("conversation must end with a user or tool message")

    def generate(self, messages: list[Message], tools: list[ToolSpec], tool_choice: ToolChoice,
                 max_output_tokens: int = 2048) -> LLMTurn:
        self.check_conversation(messages)
        self.seen.append(list(messages))
        if not self._script:
            raise ScriptError("script exhausted: the loop made more LLM calls than scripted")
        step = self._script.pop(0)
        self.calls += 1
        if isinstance(step, dict) and "error" in step:
            step = LLMError(f"scripted {step['error']} failure", transient=step["error"] == "transient")
        if isinstance(step, Exception):
            raise step
        if isinstance(step, dict):
            step = LLMTurn(text=step.get("text", ""), tool_calls=[
                ToolCall(id=f"s{self.calls}_{i}", name=c["name"], args=c.get("args", {}))
                for i, c in enumerate(step.get("tool_calls", []))])
        names = {t.name for t in tools}
        for c in step.tool_calls:
            if c.name not in names:
                raise ScriptError(f"scripted call to tool {c.name!r} that was not offered")
            if tool_choice.mode == "any" and tool_choice.allowed and c.name not in tool_choice.allowed:
                raise ScriptError(f"scripted call {c.name!r} violates tool_choice {tool_choice.allowed}")
        if tool_choice.mode == "any" and not step.tool_calls:
            raise ScriptError("tool_choice=any requires a tool call")
        return step


class FaultInjectingProvider:
    """Wraps a provider; every call with index >= fail_from_call (1-based) raises a transient
    error, so the retry fails too. Used to demo/test the degraded path in a controlled way."""

    def __init__(self, inner: Any, fail_from_call: int):
        self._inner, self._from = inner, fail_from_call
        self.name, self.model = inner.name, inner.model
        self.calls = 0

    def generate(self, messages, tools, tool_choice, max_output_tokens: int = 2048) -> LLMTurn:
        self.calls += 1
        if self.calls >= self._from:
            raise LLMError("injected provider failure", transient=True)
        return self._inner.generate(messages, tools, tool_choice, max_output_tokens)

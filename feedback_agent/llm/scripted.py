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
        conversation = [m for m in messages if m.role != "system"]
        if not conversation or conversation[0].role != "user":
            raise ScriptError("conversation must start with a user message")
        for previous, current in zip(conversation, conversation[1:]):
            if previous.role == "model" and previous.tool_calls:
                expected_ids = [call.id for call in previous.tool_calls]
                received_ids = None
                if current.role == "tool":
                    received_ids = [response.call_id for response in current.tool_responses]
                if received_ids != expected_ids:
                    raise ScriptError(f"function calls {expected_ids} must be answered by one tool message with the "
                                      f"same ids in order, got {received_ids}")
            elif current.role == "tool":
                raise ScriptError("tool message without a preceding model turn that made calls")
            elif current.role == previous.role == "model":
                raise ScriptError("two consecutive model turns")
        if conversation[-1].role == "model":
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
                ToolCall(id=f"s{self.calls}_{i}", name=call["name"], args=call.get("args", {}))
                for i, call in enumerate(step.get("tool_calls", []))])
        names = {tool.name for tool in tools}
        for call in step.tool_calls:
            if call.name not in names:
                raise ScriptError(f"scripted call to tool {call.name!r} that was not offered")
            if tool_choice.mode == "any" and tool_choice.allowed and call.name not in tool_choice.allowed:
                raise ScriptError(f"scripted call {call.name!r} violates tool_choice {tool_choice.allowed}")
        if tool_choice.mode == "any" and not step.tool_calls:
            raise ScriptError("tool_choice=any requires a tool call")
        return step


class FaultInjectingProvider:
    """Wraps a provider; every call with index >= fail_from_call (1-based) raises a transient
    error, so the retry fails too. Used to demo/test the degraded path in a controlled way."""

    def __init__(self, inner: Any, fail_from_call: int):
        self._inner, self._fail_from_call = inner, fail_from_call
        self.name, self.model = inner.name, inner.model
        self.calls = 0

    def generate(self, messages, tools, tool_choice, max_output_tokens: int = 2048) -> LLMTurn:
        self.calls += 1
        if self.calls >= self._fail_from_call:
            raise LLMError("injected provider failure", transient=True)
        return self._inner.generate(messages, tools, tool_choice, max_output_tokens)

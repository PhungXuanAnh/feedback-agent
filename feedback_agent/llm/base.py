"""Provider-neutral LLM contract. Adding a provider = one class implementing LLMProvider
plus one line in llm/__init__.py (see PROVIDERS)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol


class LLMError(RuntimeError):
    """Provider failure. `transient` errors (network, 429, 5xx) may be retried once."""

    def __init__(self, message: str, transient: bool = False, usage: Optional[dict] = None,
                 usage_unknown: bool = False):
        super().__init__(message)
        self.transient = transient
        self.usage = usage          # tokens the provider billed for this failed attempt, when it said so
        # True when tokens may have been consumed but are not known (answered without usage, or the connection
        # broke after the request was sent). False = known free: never reached the provider, or it rejected the call.
        self.usage_unknown = usage_unknown


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema


@dataclass
class ToolChoice:
    mode: Literal["auto", "any", "none"] = "auto"
    allowed: Optional[list[str]] = None  # with mode="any": only these tools may be called


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResponse:
    call_id: str
    name: str
    response: dict[str, Any]


@dataclass
class Message:
    """role: system | user | model | tool. A `tool` message carries one response per
    call of the previous model turn (provider maps it to the user role)."""

    role: Literal["system", "user", "model", "tool"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_responses: list[ToolResponse] = field(default_factory=list)
    raw: Any = None  # model turns: provider-native content, replayed verbatim


@dataclass
class LLMTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Optional[dict[str, Optional[int]]] = None  # prompt_tokens / output_tokens / total_tokens
    raw: Any = None  # includes provider-only fields such as thought signatures; never traced

    def as_message(self) -> Message:
        return Message(role="model", text=self.text, tool_calls=list(self.tool_calls), raw=self.raw)


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, messages: list[Message], tools: list[ToolSpec],
                 tool_choice: ToolChoice, max_output_tokens: int = 2048) -> LLMTurn: ...

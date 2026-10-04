"""Provider factory. To add a provider: write a class that satisfies `LLMProvider`
(llm/base.py) and add one line to PROVIDERS."""
from __future__ import annotations

from typing import Any, Callable

from ..config import Settings
from .base import LLMError, LLMProvider, LLMTurn, Message, ToolCall, ToolChoice, ToolResponse, ToolSpec
from .gemini import GeminiProvider
from .scripted import FaultInjectingProvider, ScriptedProvider, ScriptError


class ConfigError(ValueError):
    pass


def _gemini(s: Settings, **_: Any) -> LLMProvider:
    if not s.llm_api_key:
        raise ConfigError("GEMINI_API_KEY (or LLM_API_KEY) is not set; use LLM_PROVIDER=scripted for offline runs")
    if not s.llm_model:
        raise ConfigError("LLM_MODEL is not set (see .env.example)")
    return GeminiProvider(s.llm_api_key, s.llm_model, s.llm_base_url, s.llm_temperature, s.llm_timeout_s,
                          s.llm_thinking_level)


def _scripted(s: Settings, script: Any = None, **_: Any) -> LLMProvider:
    if script is None:
        raise ConfigError("the scripted provider needs a script (turns list or path to a JSON file)")
    return ScriptedProvider(script) if isinstance(script, list) else ScriptedProvider.from_file(script)


PROVIDERS: dict[str, Callable[..., LLMProvider]] = {"gemini": _gemini, "scripted": _scripted}


def get_provider(settings: Settings, **kwargs: Any) -> LLMProvider:
    try:
        make = PROVIDERS[settings.llm_provider]
    except KeyError:
        raise ConfigError(f"Unknown LLM_PROVIDER {settings.llm_provider!r}; available: {sorted(PROVIDERS)}") from None
    return make(settings, **kwargs)

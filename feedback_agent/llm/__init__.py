"""Provider factory. To add a provider: write a class that satisfies `LLMProvider`
(llm/base.py) and add one line to PROVIDERS."""

from __future__ import annotations

from typing import Any, Callable

from ..config import Settings
from .base import (
    LLMError,
    LLMProvider,
    LLMTurn,
    Message,
    ToolCall,
    ToolChoice,
    ToolResponse,
    ToolSpec,
)
from .gemini import GeminiProvider
from .scripted import FaultInjectingProvider, ScriptedProvider, ScriptError


class ConfigError(ValueError):
    pass


def _make_gemini(settings: Settings, **_: Any) -> LLMProvider:
    """Build the Gemini provider; ConfigError if the API key or the model is not set."""
    if not settings.llm_api_key:
        raise ConfigError(
            "GEMINI_API_KEY (or LLM_API_KEY) is not set; use LLM_PROVIDER=scripted for offline runs"
        )
    if not settings.llm_model:
        raise ConfigError("LLM_MODEL is not set (see .env.example)")
    return GeminiProvider(
        settings.llm_api_key,
        settings.llm_model,
        settings.llm_base_url,
        settings.llm_temperature,
        settings.llm_timeout_s,
        settings.llm_thinking_level,
    )


def _make_scripted(settings: Settings, script: Any = None, **_: Any) -> LLMProvider:
    """Build the scripted provider from a list of turns or a JSON file path."""
    if script is None:
        raise ConfigError(
            "the scripted provider needs a script (turns list or path to a JSON file)"
        )
    return (
        ScriptedProvider(script) if isinstance(script, list) else ScriptedProvider.from_file(script)
    )


PROVIDERS: dict[str, Callable[..., LLMProvider]] = {
    "gemini": _make_gemini,
    "scripted": _make_scripted,
}


def get_provider(settings: Settings, **kwargs: Any) -> LLMProvider:
    """Provider named by settings.llm_provider ("gemini" | "scripted").

    Extra kwargs go to its factory, e.g. get_provider(settings, script="samples/scripts/x.json").
    """
    try:
        factory = PROVIDERS[settings.llm_provider]
    except KeyError:
        raise ConfigError(
            f"Unknown LLM_PROVIDER {settings.llm_provider!r}; available: {sorted(PROVIDERS)}"
        ) from None
    return factory(settings, **kwargs)

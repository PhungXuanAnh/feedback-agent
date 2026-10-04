"""Settings read from the environment (and .env if present). No secrets in code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _opt_float(name: str) -> Optional[float]:
    v = os.getenv(name, "").strip()
    return float(v) if v else None


@dataclass
class Settings:
    llm_provider: str = "gemini"
    llm_model: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_base_url: str = "https://generativelanguage.googleapis.com"
    llm_temperature: Optional[float] = None
    llm_thinking_level: str = ""  # minimal | low | medium | high; empty = model default
    max_llm_turns: int = 4
    max_tool_calls: int = 6
    max_forced_lookups: int = 3
    max_output_tokens: int = 8192
    llm_retry_backoff_s: float = 0.5
    llm_timeout_s: float = 60.0
    price_in: Optional[float] = None
    price_out: Optional[float] = None
    database_path: str = str(ROOT / "data" / "feedback_agent.db")
    trace_dir: str = str(ROOT / "traces")
    data_dir: str = str(ROOT / "data")
    # API protection for a shared/public deployment; every limit is off (0 / empty) unless set
    api_token: str = field(default="", repr=False)
    rate_limit_per_min: int = 0       # POST /feedback per client IP
    daily_request_limit: int = 0      # reports created per UTC day, all clients
    daily_token_limit: int = 0        # prompt+output tokens per UTC day, all clients
    trust_proxy: bool = False         # take the client IP from X-Forwarded-For (only behind your own proxy)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env")
        e = os.getenv
        return cls(
            llm_provider=e("LLM_PROVIDER", "gemini").strip().lower(),
            llm_model=e("LLM_MODEL", "").strip(),
            llm_api_key=e("GEMINI_API_KEY") or e("LLM_API_KEY", ""),
            llm_base_url=e("LLM_BASE_URL", "https://generativelanguage.googleapis.com"),
            llm_temperature=_opt_float("LLM_TEMPERATURE"),
            llm_thinking_level=e("LLM_THINKING_LEVEL", "").strip().lower(),
            max_llm_turns=int(e("MAX_LLM_TURNS", "4")),
            max_tool_calls=int(e("MAX_TOOL_CALLS", "6")),
            max_output_tokens=int(e("MAX_OUTPUT_TOKENS", "8192")),
            price_in=_opt_float("PRICE_PER_MTOK_IN"),
            price_out=_opt_float("PRICE_PER_MTOK_OUT"),
            database_path=e("DATABASE_PATH", str(ROOT / "data" / "feedback_agent.db")),
            trace_dir=e("TRACE_DIR", str(ROOT / "traces")),
            api_token=e("API_TOKEN", ""),
            rate_limit_per_min=int(e("RATE_LIMIT_PER_MIN", "0")),
            daily_request_limit=int(e("DAILY_REQUEST_LIMIT", "0")),
            daily_token_limit=int(e("DAILY_TOKEN_LIMIT", "0")),
            trust_proxy=e("TRUST_PROXY", "").strip().lower() in ("1", "true", "yes"),
        )

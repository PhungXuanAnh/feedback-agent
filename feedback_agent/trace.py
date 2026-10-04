"""Per-request JSONL trace: one line per step, so a reader can see HOW a report was reached."""
from __future__ import annotations

import json
import re
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .config import Settings


class Event(str, Enum):
    intake = "intake"
    injection_guard = "injection_guard"
    classification_completed = "classification_completed"
    system_evidence = "system_evidence"
    llm_call = "llm_call"
    llm_error = "llm_error"
    tool_call = "tool_call"
    tool_result = "tool_result"
    tool_error = "tool_error"
    grounding_validation = "grounding_validation"
    degraded_path = "degraded_path"
    report_generated = "report_generated"
    human_review = "human_review"


def new_trace_id() -> str:
    return "tr_" + uuid.uuid4().hex[:12]


class Tracer:
    def __init__(self, trace_id: str, settings: Settings, enabled: bool = True):
        self.trace_id = trace_id
        self.events: list[dict[str, Any]] = []
        self._price_in, self._price_out = settings.price_in, settings.price_out
        self._path: Optional[Path] = None
        self._seq0 = 0
        if enabled and settings.trace_dir:
            Path(settings.trace_dir).mkdir(parents=True, exist_ok=True)
            self._path = Path(settings.trace_dir) / f"{trace_id}.jsonl"
            if self._path.exists():  # continuing an existing trace (e.g. the human review step)
                self._seq0 = sum(1 for _ in open(self._path, encoding="utf-8"))
        self.llm_calls = 0
        self.provider_calls = 0
        self._usage_missing = 0
        self._cost_sum = 0.0
        self.prompt_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self._t0 = time.monotonic()

    def emit(self, event: Event, **fields: Any) -> None:
        rec = {"seq": self._seq0 + len(self.events) + 1, "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
               "trace_id": self.trace_id, "event": event.value, **fields}
        self.events.append(rec)
        if self._path:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")

    def cost_of(self, usage: Optional[dict]) -> Any:
        """Estimated USD; only when prices are configured AND the provider reported usage."""
        if not usage or usage.get("prompt_tokens") is None or self._price_in is None or self._price_out is None:
            return "unknown"
        return round((usage["prompt_tokens"] * self._price_in + (usage.get("output_tokens") or 0)
                      * self._price_out) / 1e6, 6)

    def _account(self, usage: Optional[dict], unknown: bool) -> Any:
        """One provider attempt (successful or failed). Returns this attempt's cost: a number or "unknown"."""
        self.provider_calls += 1
        if usage and usage.get("prompt_tokens") is not None:
            self.prompt_tokens = (self.prompt_tokens or 0) + usage["prompt_tokens"]
            self.output_tokens = (self.output_tokens or 0) + (usage.get("output_tokens") or 0)
            cost = self.cost_of(usage)
            if cost != "unknown":
                self._cost_sum += cost
            return cost
        if unknown:  # tokens may have been consumed but are not known: the totals below are a lower bound
            self._usage_missing += 1
        return "unknown"

    def record_llm_call(self, usage: Optional[dict], **fields: Any) -> None:
        self.llm_calls += 1
        cost = self._account(usage, True)
        self.emit(Event.llm_call, usage=usage, cost_usd=cost, **fields)

    def record_failed_attempt(self, usage: Optional[dict], unknown: bool, **fields: Any) -> None:
        cost = self._account(usage, unknown)
        self.emit(Event.llm_error, usage=usage, cost_usd=cost, **fields)

    def totals(self) -> dict[str, Any]:
        """llm_calls = successful provider calls; provider_calls = every attempt incl. retries and failures.
        usage_complete covers the attempts that COULD have been billed: it is False when any such attempt has unknown
        usage (answered without usage, or the connection broke after sending); tokens and cost are then a lower bound.
        Attempts that never reached the provider or were rejected with an HTTP error count as known-free."""
        priced = self._price_in is not None and self._price_out is not None and self.prompt_tokens is not None
        return {"llm_calls": self.llm_calls, "provider_calls": self.provider_calls,
                "prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens,
                "cost_usd": round(self._cost_sum, 6) if priced else "unknown",
                "usage_complete": self.provider_calls > 0 and self._usage_missing == 0,
                "elapsed_ms": int((time.monotonic() - self._t0) * 1000)}


def load_trace_by_id(trace_id: str, trace_dir: str) -> list[dict]:
    """API-safe lookup: only a trace id (tr_ + 12 hex) inside trace_dir, never a path."""
    if not re.fullmatch(r"tr_[0-9a-f]{12}", trace_id):
        raise FileNotFoundError(trace_id)
    return load_trace(str(Path(trace_dir) / f"{trace_id}.jsonl"), trace_dir)


def load_trace(path_or_id: str, trace_dir: str) -> list[dict]:
    p = Path(path_or_id)
    if not p.exists():
        p = Path(trace_dir) / f"{path_or_id}.jsonl"
    return [json.loads(line) for line in p.read_text("utf-8").splitlines() if line.strip()]


def render_trace(events: list[dict]) -> str:
    """Human-readable trace: one block per event."""
    out = []
    for e in events:
        extra = {k: v for k, v in e.items() if k not in ("seq", "ts", "trace_id", "event")}
        body = json.dumps(extra, ensure_ascii=False, default=str)
        out.append(f"#{e['seq']:02d} {e['event']:<24} {body if len(body) < 400 else body[:400] + ' ...'}")
    head = f"trace {events[0]['trace_id']}" if events else "empty trace"
    return head + "\n" + "\n".join(out)

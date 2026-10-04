"""The three mandatory retrieval tools (CS guidelines, customer record, company policy).

Every tool returns one of three states:
  found      - record(s) returned
  not_found  - the source was consulted and has no matching record (counts as "consulted")
  error      - the source failed or the call was refused (does NOT count as consulted)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from .llm.base import ToolSpec
from .schemas import Category, TIER_ORDER
from .store import Store


class ToolStatus(str, Enum):
    found = "found"
    not_found = "not_found"
    error = "error"


class TransientToolError(RuntimeError):
    """A source hiccup that is worth exactly one retry."""


@dataclass
class ToolResult:
    tool: str
    source: str
    status: ToolStatus
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None
    retryable: bool = False
    flags: list[str] = field(default_factory=list)

    @property
    def consulted(self) -> bool:
        return self.status in (ToolStatus.found, ToolStatus.not_found)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"status": self.status.value}
        if self.data is not None:
            d["data"] = self.data
        if self.error:
            d["error"] = self.error
        if self.message:
            d["message"] = self.message
        return d


SOURCES = {"get_cs_guidelines": "guidelines", "lookup_customer": "customer", "search_policies": "policies"}
RETRIEVAL_TOOLS = list(SOURCES)


# ------------------------------------------------------------------ arguments
class GuidelinesArgs(BaseModel):
    category: Category = Field(description="Feedback category to fetch the standard process for.")


class CustomerArgs(BaseModel):
    email: Optional[str] = Field(default=None, description="Customer email from the feedback metadata.")
    customer_id: Optional[str] = Field(default=None, description="Customer id, only if known from metadata.")


class PolicyArgs(BaseModel):
    query: str = Field(min_length=2, max_length=200,
                       description="Short keyword query describing the policy question, e.g. 'duplicate charge refund'.")
    category: Optional[Category] = Field(default=None, description="Optional category; only boosts ranking.")
    top_k: int = Field(default=3, ge=1, le=3, description="Number of policies to return (max 3).")


ARG_MODELS = {"get_cs_guidelines": GuidelinesArgs, "lookup_customer": CustomerArgs, "search_policies": PolicyArgs}

TOOL_DESCRIPTIONS = {
    "get_cs_guidelines": (
        "Retrieve the standard customer-support process (CS workflow guideline) for ONE feedback category. "
        "Returns the guideline id, steps and which action types it supports. Call it for the classified "
        "category; for ambiguous feedback you may also call it for an alternative category."),
    "lookup_customer": (
        "Look up the customer who sent this feedback: tier, tenure, open tickets, recent invoices and "
        "duplicate-charge candidates. You can only look up the sender of THIS feedback (the email or id in "
        "the metadata); anything else is refused as out_of_scope. status=not_found means no record exists "
        "for the sender: treat the customer as unverified and never invent a tier."),
    "search_policies": (
        "Search the company policies in force on the feedback date (refunds, SLA, escalation, security, "
        "privacy, conduct). Returns at most 3 policies with their conditions (tier, max amount, time window) "
        "and which action types they allow. Use short keywords; if nothing matches the result is not_found "
        "and you must say that no policy applies."),
}


def tool_specs(include_submit: Any = None) -> list[ToolSpec]:
    specs = [ToolSpec(n, TOOL_DESCRIPTIONS[n], ARG_MODELS[n].model_json_schema()) for n in RETRIEVAL_TOOLS]
    if include_submit is not None:
        specs.append(include_submit)
    return specs


# ------------------------------------------------------------- knowledge base
_STOP = {"the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "is", "are", "was", "be", "by", "with",
         "it", "this", "that", "at", "as", "from", "customer", "customers", "any", "may", "must"}


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def _tokens(text: str) -> set[str]:
    return {_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOP}


MIN_SCORE = 2


class KnowledgeBase:
    """File-backed guidelines + policies (JSON). Swap for a real KB/MCP behind the same methods."""

    def __init__(self, guidelines: list[dict], policies: list[dict]):
        self.guidelines = guidelines
        self.policies = policies

    @classmethod
    def load(cls, data_dir: str | Path) -> "KnowledgeBase":
        d = Path(data_dir)
        return cls(json.loads((d / "guidelines.json").read_text("utf-8")),
                   json.loads((d / "policies.json").read_text("utf-8")))

    def guideline_for(self, category: str) -> Optional[dict]:
        return next((g for g in self.guidelines if g["category"] == category), None)

    def guideline_by_id(self, gid: str) -> Optional[dict]:
        return next((g for g in self.guidelines if g["guideline_id"] == gid), None)

    def active_policies(self, as_of: date) -> list[dict]:
        """In force on `as_of`: effective_date <= as_of < expires_date, and not superseded by a policy
        that is itself in force (even when the old one has no expiry)."""
        live = [p for p in self.policies
                if date.fromisoformat(p["effective_date"]) <= as_of
                and (p["expires_date"] is None or as_of < date.fromisoformat(p["expires_date"]))]
        replaced = {old for p in live for old in p.get("supersedes", [])}
        return [p for p in live if p["policy_id"] not in replaced]

    def search_policies(self, query: str, category: Optional[str], top_k: int, as_of: date) -> list[dict]:
        q = _tokens(query)
        scored = []
        for p in self.active_policies(as_of):
            kw = set().union(*[_tokens(k) for k in p["keywords"]]) if p["keywords"] else set()
            text = _tokens(p["title"] + " " + p["text"]) - kw
            score = 2 * len(q & kw) + len(q & text)
            if category and category in p["categories"]:
                score += 3  # boost only, never a filter
            if score >= MIN_SCORE:
                scored.append((score, p))
        scored.sort(key=lambda sp: (-sp[0], sp[1]["policy_id"]))
        return [dict(p, score=s) for s, p in scored[:top_k]]


# ----------------------------------------------------------------- run context
@dataclass
class FaultInjector:
    """Controlled failures for demos/tests. tool_failures maps tool name -> number of
    transient failures to raise (-1 = always)."""

    tool_failures: dict[str, int] = field(default_factory=dict)

    def check(self, tool: str) -> None:
        left = self.tool_failures.get(tool, 0)
        if left != 0:
            if left > 0:
                self.tool_failures[tool] = left - 1
            raise TransientToolError(f"injected failure in {tool}")


@dataclass
class ToolContext:
    store: Store
    kb: KnowledgeBase
    metadata_email: str
    metadata_customer_id: Optional[str]
    as_of: date
    faults: FaultInjector = field(default_factory=FaultInjector)


def _months_between(start: date, end: date) -> int:
    return max(0, (end.year - start.year) * 12 + end.month - start.month - (1 if end.day < start.day else 0))


def duplicate_charge_candidates(invoices: list[dict], max_gap_days: int = 3) -> list[dict]:
    """Paid invoices with the same amount at most `max_gap_days` apart (code-computed, not LLM)."""
    paid = sorted((i for i in invoices if i["status"] == "paid"), key=lambda i: (i["amount"], i["date"]))
    out, used = [], set()
    for i, a in enumerate(paid):
        if a["invoice_id"] in used:
            continue
        group = [a]
        for b in paid[i + 1:]:
            gap = abs((date.fromisoformat(b["date"]) - date.fromisoformat(a["date"])).days)
            if b["amount"] == a["amount"] and gap <= max_gap_days and b["invoice_id"] not in used:
                group.append(b)
        if len(group) > 1:
            used.update(g["invoice_id"] for g in group)
            out.append({"invoice_ids": [g["invoice_id"] for g in group], "amount": a["amount"],
                        "dates": sorted({g["date"] for g in group})})
    return out


# ------------------------------------------------------------------ tool bodies
def _get_cs_guidelines(ctx: ToolContext, a: GuidelinesArgs) -> ToolResult:
    g = ctx.kb.guideline_for(a.category.value)
    if not g:
        return ToolResult("get_cs_guidelines", "guidelines", ToolStatus.not_found,
                          message=f"No guideline for category {a.category.value}")
    return ToolResult("get_cs_guidelines", "guidelines", ToolStatus.found, data=g)


def _lookup_customer(ctx: ToolContext, a: CustomerArgs) -> ToolResult:
    tool, src = "lookup_customer", "customer"
    if not a.email and not a.customer_id:
        return ToolResult(tool, src, ToolStatus.error, error="invalid_arguments",
                          message="Provide email or customer_id from the feedback metadata.")
    record = ctx.store.customer_by_email(ctx.metadata_email)
    # Scope lock: the lookup may only target the sender named in the feedback metadata.
    if a.email and a.email.strip().lower() != ctx.metadata_email:
        return ToolResult(tool, src, ToolStatus.error, error="out_of_scope", flags=["scope_violation"],
                          message="Only the sender of this feedback can be looked up.")
    if a.customer_id and (record is None or a.customer_id != record["id"]):
        return ToolResult(tool, src, ToolStatus.error, error="out_of_scope", flags=["scope_violation"],
                          message="Only the sender of this feedback can be looked up.")
    if record is None:
        return ToolResult(tool, src, ToolStatus.not_found,
                          message="No customer record for this sender; treat as unverified.")
    tickets = ctx.store.tickets_for(record["id"])
    invoices = ctx.store.invoices_for(record["id"])
    data = {
        "customer_id": record["id"], "name": record["name"], "email": record["email"], "tier": record["tier"],
        "signup_date": record["signup_date"],
        "tenure_months": _months_between(date.fromisoformat(record["signup_date"]), ctx.as_of),
        "open_tickets": [t for t in tickets if t["status"] == "open"],
        "recent_invoices": invoices[:6],
        "duplicate_charge_candidates": duplicate_charge_candidates(invoices),
    }
    return ToolResult(tool, src, ToolStatus.found, data=data)


def _search_policies(ctx: ToolContext, a: PolicyArgs) -> ToolResult:
    hits = ctx.kb.search_policies(a.query, a.category.value if a.category else None, a.top_k, ctx.as_of)
    if not hits:
        return ToolResult("search_policies", "policies", ToolStatus.not_found,
                          data={"as_of": ctx.as_of.isoformat()},
                          message="No policy in force on the feedback date matched the query.")
    return ToolResult("search_policies", "policies", ToolStatus.found,
                      data={"as_of": ctx.as_of.isoformat(), "policies": hits})


_BODIES = {"get_cs_guidelines": _get_cs_guidelines, "lookup_customer": _lookup_customer,
           "search_policies": _search_policies}


def run_tool(ctx: ToolContext, name: str, raw_args: dict[str, Any]) -> ToolResult:
    """Single attempt. Validates args (allow-list), injects faults, converts failures to `error`."""
    if name not in _BODIES:
        return ToolResult(name, "unknown", ToolStatus.error, error="unknown_tool")
    src = SOURCES[name]
    try:
        args = ARG_MODELS[name].model_validate(raw_args or {})
    except ValidationError as e:
        msg = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())[:300]
        return ToolResult(name, src, ToolStatus.error, error="invalid_arguments", message=msg)
    try:
        ctx.faults.check(name)
        return _BODIES[name](ctx, args)
    except TransientToolError as e:
        return ToolResult(name, src, ToolStatus.error, error="source_unavailable", message=str(e), retryable=True)
    except Exception as e:  # a broken source must never crash the run
        return ToolResult(name, src, ToolStatus.error, error="source_failure", message=type(e).__name__)


def tier_rank(tier: Optional[str]) -> int:
    return TIER_ORDER.index(tier) if tier in TIER_ORDER else -1

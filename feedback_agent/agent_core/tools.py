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

from ..database import Database
from ..llm.base import ToolSpec
from ..schemas import Category, ReportDraft, TIER_ORDER


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
        """found or not_found: the source answered, even if it had no record."""
        return self.status in (ToolStatus.found, ToolStatus.not_found)

    def to_dict(self) -> dict[str, Any]:
        """What the LLM sees of the result: status, plus data/error/message when set."""
        d: dict[str, Any] = {"status": self.status.value}
        if self.data is not None:
            d["data"] = self.data
        if self.error:
            d["error"] = self.error
        if self.message:
            d["message"] = self.message
        return d


SOURCES = {
    "get_cs_guidelines": "guidelines",
    "lookup_customer": "customer",
    "search_policies": "policies",
}
RETRIEVAL_TOOLS = list(SOURCES)


# ------------------------------------------------------------------ arguments
class GuidelinesArgs(BaseModel):
    category: Category = Field(description="Feedback category to fetch the standard process for.")


class CustomerArgs(BaseModel):
    email: Optional[str] = Field(
        default=None, description="Customer email from the feedback metadata."
    )
    customer_id: Optional[str] = Field(
        default=None, description="Customer id, only if known from metadata."
    )


class PolicyArgs(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=200,
        description="Short keyword query describing the policy question, e.g. 'duplicate charge refund'.",
    )
    category: Optional[Category] = Field(
        default=None, description="Optional category; only boosts ranking."
    )
    top_k: int = Field(default=3, ge=1, le=3, description="Number of policies to return (max 3).")


RETRIEVAL_ARG_MODELS = {
    "get_cs_guidelines": GuidelinesArgs,
    "lookup_customer": CustomerArgs,
    "search_policies": PolicyArgs,
}

SUBMIT_TOOL = ToolSpec(
    "submit_report",
    "Submit the final report for the CS officer. Allowed only after guidelines, customer and policies "
    "were all consulted. The system validates every citation and action against the tool results; if "
    "it is rejected, fix the listed problems and call it again.",
    ReportDraft.model_json_schema(),
)

# Offered on every agent turn except the last, which offers only SUBMIT_TOOL.
AGENT_TOOLS = [
    ToolSpec(
        "get_cs_guidelines",
        "Retrieve the standard customer-support process (CS workflow guideline) for ONE feedback category. "
        "Returns the guideline id, steps and which action types it supports. Call it for the classified "
        "category; for ambiguous feedback you may also call it for an alternative category.",
        GuidelinesArgs.model_json_schema(),
    ),
    ToolSpec(
        "lookup_customer",
        "Look up the customer who sent this feedback: tier, tenure, open tickets, recent invoices and "
        "duplicate-charge candidates. You can only look up the sender of THIS feedback (the email or id in "
        "the metadata); anything else is refused as out_of_scope. status=not_found means no record exists "
        "for the sender: treat the customer as unverified and never invent a tier.",
        CustomerArgs.model_json_schema(),
    ),
    ToolSpec(
        "search_policies",
        "Search the company policies in force on the feedback date (refunds, SLA, escalation, security, "
        "privacy, conduct). Returns at most 3 policies with their conditions (tier, max amount, time window) "
        "and which action types they allow. Use short keywords; if nothing matches the result is not_found "
        "and you must say that no policy applies.",
        PolicyArgs.model_json_schema(),
    ),
    SUBMIT_TOOL,
]


# ------------------------------------------------------------- knowledge base
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "and",
    "or",
    "in",
    "on",
    "is",
    "are",
    "was",
    "be",
    "by",
    "with",
    "it",
    "this",
    "that",
    "at",
    "as",
    "from",
    "customer",
    "customers",
    "any",
    "may",
    "must",
}


def _stem(token: str) -> str:
    """Tiny stemmer: drop a plural "s" ("refunds" -> "refund"; short words like "bus" stay)."""
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _tokens(text: str) -> set[str]:
    """Stemmed words minus stop words: "Refunds for the invoice" -> {"refund", "invoice"}."""
    return {
        _stem(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOP
    }


MIN_SCORE = 2


class KnowledgeBase:
    """File-backed guidelines + policies (JSON). Swap for a real KB/MCP behind the same methods."""

    def __init__(self, guidelines: list[dict], policies: list[dict]):
        self.guidelines = guidelines
        self.policies = policies

    @classmethod
    def load(cls, data_dir: str | Path) -> "KnowledgeBase":
        """Load guidelines.json and policies.json from `data_dir`."""
        directory = Path(data_dir)
        return cls(
            json.loads((directory / "guidelines.json").read_text("utf-8")),
            json.loads((directory / "policies.json").read_text("utf-8")),
        )

    def guideline_for_category(self, category: str) -> Optional[dict]:
        """The CS guideline of a category (one per category), or None."""
        for guideline in self.guidelines:
            if guideline["category"] == category:
                return guideline
        return None

    def guideline_by_id(self, guideline_id: str) -> Optional[dict]:
        """Guideline with this id, e.g. "GL-TRIAGE" (added by the pipeline as system evidence)."""
        for guideline in self.guidelines:
            if guideline["guideline_id"] == guideline_id:
                return guideline
        return None

    def active_policies(self, as_of: date) -> list[dict]:
        """In force on `as_of`: effective_date <= as_of < expires_date, and not superseded by a policy
        that is itself in force (even when the old one has no expiry)."""
        in_force = [
            policy
            for policy in self.policies
            if date.fromisoformat(policy["effective_date"]) <= as_of
            and (
                policy["expires_date"] is None or as_of < date.fromisoformat(policy["expires_date"])
            )
        ]
        replaced = {old_id for policy in in_force for old_id in policy.get("supersedes", [])}
        return [policy for policy in in_force if policy["policy_id"] not in replaced]

    def search_policies(
        self, query: str, category: Optional[str], top_k: int, as_of: date
    ) -> list[dict]:
        """Keyword search over policies in force on `as_of`: best `top_k` first, each with a score.

        Score = 2 per query word found in the policy keywords + 1 per word found in title/text,
        +3 if the category matches. A policy the query does not match at all is dropped, whatever
        its category, and so is one scoring under MIN_SCORE (2).
        """
        query_tokens = _tokens(query)
        scored = []
        for policy in self.active_policies(as_of):
            keyword_tokens = set()
            for keyword in policy["keywords"]:
                keyword_tokens |= _tokens(keyword)
            text_tokens = _tokens(policy["title"] + " " + policy["text"]) - keyword_tokens
            score = 2 * len(query_tokens & keyword_tokens) + len(query_tokens & text_tokens)
            if score == 0:
                continue
            if category and category in policy["categories"]:
                score += 3  # boost only, never a filter
            if score >= MIN_SCORE:
                scored.append((score, policy))
        scored.sort(key=lambda scored_policy: (-scored_policy[0], scored_policy[1]["policy_id"]))
        return [dict(policy, score=policy_score) for policy_score, policy in scored[:top_k]]


# ----------------------------------------------------------------- run context
@dataclass
class FaultInjector:
    """Controlled failures for demos/tests. tool_failures maps tool name -> number of
    transient failures to raise (-1 = always)."""

    tool_failures: dict[str, int] = field(default_factory=dict)

    def fail_if_scheduled(self, tool_name: str) -> None:
        """Raise TransientToolError if a failure is scheduled for this tool, and use it up.

        Example: {"search_policies": 1} fails only the first call; -1 fails every call.
        """
        failures_left = self.tool_failures.get(tool_name, 0)
        if failures_left != 0:
            if failures_left > 0:
                self.tool_failures[tool_name] = failures_left - 1
            raise TransientToolError(f"injected failure in {tool_name}")


@dataclass
class ToolContext:
    database: Database
    knowledge_base: KnowledgeBase
    metadata_email: str
    metadata_customer_id: Optional[str]
    as_of: date
    fault_injector: FaultInjector = field(default_factory=FaultInjector)


def _months_between(start: date, end: date) -> int:
    """Whole months from start to end, never negative: 2026-01-31 -> 2026-03-30 is 1."""
    return max(
        0,
        (end.year - start.year) * 12 + end.month - start.month - (1 if end.day < start.day else 0),
    )


def duplicate_charge_candidates(invoices: list[dict], max_gap_days: int = 3) -> list[dict]:
    """Paid invoices with the same amount at most `max_gap_days` apart (code-computed, not LLM).

    Example: two paid 49.00 invoices on 06-01 and 06-02 -> one group {"invoice_ids": [..2 ids..],
    "amount": 49.0, "dates": [...]}.
    """
    paid = sorted(
        (i for i in invoices if i["status"] == "paid"), key=lambda i: (i["amount"], i["date"])
    )
    groups, grouped_ids = [], set()
    for i, first in enumerate(paid):
        if first["invoice_id"] in grouped_ids:
            continue
        group = [first]
        for other in paid[i + 1 :]:
            gap = abs((date.fromisoformat(other["date"]) - date.fromisoformat(first["date"])).days)
            if (
                other["amount"] == first["amount"]
                and gap <= max_gap_days
                and other["invoice_id"] not in grouped_ids
            ):
                group.append(other)
        if len(group) > 1:
            grouped_ids.update(invoice["invoice_id"] for invoice in group)
            groups.append(
                {
                    "invoice_ids": [invoice["invoice_id"] for invoice in group],
                    "amount": first["amount"],
                    "dates": sorted({invoice["date"] for invoice in group}),
                }
            )
    return groups


# ------------------------------------------------------------------ tool bodies
def _get_cs_guidelines(tool_context: ToolContext, args: GuidelinesArgs) -> ToolResult:
    """Tool body: the guideline of the requested category, or not_found."""
    guideline = tool_context.knowledge_base.guideline_for_category(args.category.value)
    if not guideline:
        return ToolResult(
            "get_cs_guidelines",
            "guidelines",
            ToolStatus.not_found,
            message=f"No guideline for category {args.category.value}",
        )
    return ToolResult("get_cs_guidelines", "guidelines", ToolStatus.found, data=guideline)


def _lookup_customer(tool_context: ToolContext, args: CustomerArgs) -> ToolResult:
    """Tool body: the sender's profile, open tickets, last 6 invoices and duplicate-charge groups.

    Only the sender named in the feedback metadata may be looked up; any other email or id is
    refused as out_of_scope with the scope_violation flag.
    """
    tool_name, source = "lookup_customer", "customer"
    if not args.email and not args.customer_id:
        return ToolResult(
            tool_name,
            source,
            ToolStatus.error,
            error="invalid_arguments",
            message="Provide email or customer_id from the feedback metadata.",
        )
    record = tool_context.database.customer_by_email(tool_context.metadata_email)
    # Scope lock: the lookup may only target the sender named in the feedback metadata.
    if args.email and args.email.strip().lower() != tool_context.metadata_email:
        return ToolResult(
            tool_name,
            source,
            ToolStatus.error,
            error="out_of_scope",
            flags=["scope_violation"],
            message="Only the sender of this feedback can be looked up.",
        )
    if args.customer_id and (
        record is None or args.customer_id.strip().lower() != record["id"].lower()
    ):
        return ToolResult(
            tool_name,
            source,
            ToolStatus.error,
            error="out_of_scope",
            flags=["scope_violation"],
            message="Only the sender of this feedback can be looked up.",
        )
    if record is None:
        return ToolResult(
            tool_name,
            source,
            ToolStatus.not_found,
            message="No customer record for this sender; treat as unverified.",
        )
    tickets = tool_context.database.tickets_for(record["id"])
    invoices = tool_context.database.invoices_for(record["id"])
    data = {
        "customer_id": record["id"],
        "name": record["name"],
        "email": record["email"],
        "tier": record["tier"],
        "signup_date": record["signup_date"],
        "tenure_months": _months_between(
            date.fromisoformat(record["signup_date"]), tool_context.as_of
        ),
        "open_tickets": [ticket for ticket in tickets if ticket["status"] == "open"],
        "recent_invoices": invoices[:6],
        "duplicate_charge_candidates": duplicate_charge_candidates(invoices),
    }
    return ToolResult(tool_name, source, ToolStatus.found, data=data)


def _search_policies(tool_context: ToolContext, args: PolicyArgs) -> ToolResult:
    """Tool body: policies in force on the feedback date that match the query, or not_found."""
    category = args.category.value if args.category else None
    hits = tool_context.knowledge_base.search_policies(
        args.query, category, args.top_k, tool_context.as_of
    )
    if not hits:
        return ToolResult(
            "search_policies",
            "policies",
            ToolStatus.not_found,
            data={"as_of": tool_context.as_of.isoformat()},
            message="No policy in force on the feedback date matched the query.",
        )
    return ToolResult(
        "search_policies",
        "policies",
        ToolStatus.found,
        data={"as_of": tool_context.as_of.isoformat(), "policies": hits},
    )


_BODIES = {
    "get_cs_guidelines": _get_cs_guidelines,
    "lookup_customer": _lookup_customer,
    "search_policies": _search_policies,
}


def run_tool(tool_context: ToolContext, name: str, raw_args: dict[str, Any]) -> ToolResult:
    """Run one tool once: validate args, apply injected faults, turn any failure into an error.

    Never raises: unknown tool -> unknown_tool, bad args -> invalid_arguments, transient fault ->
    source_unavailable (retryable), any other exception -> source_failure.
    """
    if name not in _BODIES:
        return ToolResult(name, "unknown", ToolStatus.error, error="unknown_tool")
    source = SOURCES[name]
    try:
        args = RETRIEVAL_ARG_MODELS[name].model_validate(raw_args or {})
    except ValidationError as e:
        error_message = "; ".join(
            f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()
        )[:300]
        return ToolResult(
            name, source, ToolStatus.error, error="invalid_arguments", message=error_message
        )
    try:
        tool_context.fault_injector.fail_if_scheduled(name)
        return _BODIES[name](tool_context, args)
    except TransientToolError as e:
        return ToolResult(
            name,
            source,
            ToolStatus.error,
            error="source_unavailable",
            message=str(e),
            retryable=True,
        )
    except Exception as e:  # a broken source must never crash the run
        return ToolResult(
            name, source, ToolStatus.error, error="source_failure", message=type(e).__name__
        )


def tier_rank(tier: Optional[str]) -> int:
    """Position of a tier in TIER_ORDER (standard=0, pro=1, enterprise=2); unknown -> -1."""
    return TIER_ORDER.index(tier) if tier in TIER_ORDER else -1

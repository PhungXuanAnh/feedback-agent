"""Grounding validator (code, not LLM): a report may only rely on records the tools returned in
THIS run. It checks that claims have a basis; it cannot check that a conclusion is right, which is
what the human officer is for."""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from ..schemas import ActionType, ReportDraft, SuggestedAction
from .evidence import Evidence
from .tools import tier_rank

_MONEY = re.compile(r"(?:\$\s?(\d[\d,]*(?:\.\d+)?))|(?:(\d[\d,]*(?:\.\d+)?)\s?(?:USD|usd|dollars))")
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _normalize_number(number: str) -> str:
    """Number text for comparison: "1,249.50" -> "1249.5", "49.0" and "49.00" -> "49"."""
    number = number.replace(",", "")
    return (
        number[:-2]
        if number.endswith(".0")
        else (number.rstrip("0").rstrip(".") if "." in number else number)
    )


# Money figures the CUSTOMER wrote. Heuristic by design (no language understanding): a number counts as
# money when it carries a currency marker ($N, N USD) or follows a charge/payment word. A number followed by a
# time unit ("37 days", "37.5 days", "1,250 days"), part of a date ("2026-09-10") or glued to an id ("INV-5005")
# never counts. The number must be matched whole: the lookahead forbids stopping before ".5" or ",250", so
# backtracking cannot turn "37.5 days" into the money figure 37.
_TIME_UNIT = r"(?:seconds?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|yrs?|%)"
_CHARGE_WORD = r"(?:charged|billed|paid|pay|refund(?:ed)?|costs?|fee|amount|price|total|owed?)"
_AFTER_CHARGE_WORD = re.compile(
    rf"\b{_CHARGE_WORD}\b[^\d$\n]{{0,25}}?(?<![A-Za-z0-9-])(\d[\d,]*(?:\.\d+)?)(?![\d-]|[.,]\d)(?!\s*{_TIME_UNIT}\b)",
    re.I,
)


def money_in_feedback(text: str) -> set[str]:
    """Money amounts the customer wrote, normalised (rules in the comment above _TIME_UNIT).

    Example: "charged $49.00 twice, 37.5 days ago, INV-5005, paid 1,249.50" -> {"49", "1249.5"}.
    """
    marked = {
        _normalize_number(match.group(1) or match.group(2)) for match in _MONEY.finditer(text)
    }
    return marked | {
        _normalize_number(match.group(1)) for match in _AFTER_CHARGE_WORD.finditer(text)
    }


def check_summary(summary: str, feedback_text: str, evidence: Evidence) -> list[str]:
    """Problems with money figures and dates in the summary ([] = none).

    An amount ($N, N USD) must be a money figure of the feedback, an invoice amount or a policy
    cap; an ISO date must appear in the feedback or a tool result. What the sentence CLAIMS
    about them is not checked.
    """
    known = money_in_feedback(feedback_text) | evidence.money_values()
    issues = []
    for match in _MONEY.finditer(summary):
        if _normalize_number(match.group(1) or match.group(2)) not in known:
            issues.append(
                f"summary mentions amount {match.group(0).strip()!r} that is not a money amount in the "
                "feedback, an invoice of this customer or a policy limit"
            )
    source = feedback_text + " " + evidence.as_text()
    for date_text in _ISO_DATE.findall(summary):
        if date_text not in source:
            issues.append(
                f"summary mentions date {date_text} that is in neither the feedback nor a tool result"
            )
    return issues


def _refund_issues(action: SuggestedAction, evidence: Evidence) -> list[str]:
    """issue_refund is the only money action: every condition is checked against run evidence.

    Needs a cited refund policy, a verified customer, a paid invoice of that customer and an
    amount <= the invoice; then at least one cited policy must pass its own limits (max_amount,
    window_days, requires_tier, requires_duplicate_candidate). [] = allowed.
    """
    refund_policies = []
    for basis_id in action.basis:
        policy = evidence.policy(basis_id)
        if policy and ActionType.issue_refund.value in policy["allowed_actions"]:
            refund_policies.append(policy)
    if not refund_policies:
        return [
            "issue_refund needs a POLICY in basis whose allowed_actions include issue_refund (a guideline is not enough)"
        ]
    if not evidence.customer_found:
        return [
            "issue_refund needs a verified customer record (lookup_customer returned not_found)"
        ]
    params = action.params
    invoice = None
    for candidate in evidence.customer["recent_invoices"]:
        if candidate["invoice_id"] == params.invoice_id:
            invoice = candidate
            break
    if invoice is None:
        return [
            f"issue_refund invoice {params.invoice_id!r} is not an invoice of this customer in the lookup_customer result"
        ]
    if params.amount is None or params.amount <= 0 or params.amount > invoice["amount"]:
        return [f"issue_refund amount must be > 0 and <= invoice amount {invoice['amount']}"]
    if invoice["status"] != "paid":
        return [
            f"issue_refund invoice {invoice['invoice_id']} is not paid (status={invoice['status']})"
        ]
    invoice_age_days = (evidence.as_of - date.fromisoformat(invoice["date"])).days
    dup_ids = {
        invoice_id
        for group in evidence.customer["duplicate_charge_candidates"]
        for invoice_id in group["invoice_ids"]
    }
    per_policy: list[str] = []
    for (
        policy
    ) in (
        refund_policies
    ):  # valid if at least one cited refund policy has all its conditions satisfied
        problems = []
        if policy.get("max_amount") is not None and params.amount > policy["max_amount"]:
            problems.append(f"amount {params.amount} exceeds max_amount {policy['max_amount']}")
        if (
            policy.get("window_days") is not None
            and not 0 <= invoice_age_days <= policy["window_days"]
        ):
            problems.append(
                f"invoice is {invoice_age_days} days old, window is {policy['window_days']} days"
            )
        if policy.get("requires_tier") and tier_rank(evidence.customer["tier"]) < tier_rank(
            policy["requires_tier"]
        ):
            problems.append(
                f"tier {evidence.customer['tier']} is below required tier {policy['requires_tier']}"
            )
        if policy.get("requires_duplicate_candidate") and invoice["invoice_id"] not in dup_ids:
            problems.append("invoice is not one of the duplicate_charge_candidates")
        if not problems:
            return []
        per_policy.append(f"{policy['policy_id']}: " + "; ".join(problems))
    return [
        "issue_refund conditions not met -> "
        + " | ".join(per_policy)
        + " (if you cannot show them, propose request_more_info or escalate_to_human instead)"
    ]


def validate_actions(
    actions: list[SuggestedAction], evidence: Evidence, clarification: Optional[str]
) -> list[str]:
    """Problems in the suggested actions ([] = all grounded).

    Each action needs basis ids from this run that allow its type; at most one refund per
    report; request_more_info needs a clarification question. Used for the LLM's draft AND for
    an officer's override.
    """
    issues: list[str] = []
    n_refunds = sum(1 for action in actions if action.type is ActionType.issue_refund)
    if n_refunds > 1:
        # Policy limits such as "up to N USD per request" apply to the whole report, not to each action, and
        # two refunds of one invoice would pay it twice. Keep it simple: one refund per report, the officer
        # handles any further invoices (same rule for the machine's draft and for an override).
        issues.append(
            f"{n_refunds} issue_refund actions in one report; propose at most ONE refund per report "
            "(mention other invoices in the summary and use escalate_to_human or request_more_info)"
        )
    for n, action in enumerate(actions, 1):
        tag = f"action {n} ({action.type.value})"
        if not action.basis:
            issues.append(f"{tag}: basis is empty; cite guideline/policy ids from tool results")
            continue
        unknown = [basis_id for basis_id in action.basis if evidence.record_by_id(basis_id) is None]
        if unknown:
            issues.append(f"{tag}: basis ids not found in this run's tool results: {unknown}")
        if not any(
            (record := evidence.record_by_id(basis_id))
            and action.type.value in record["allowed_actions"]
            for basis_id in action.basis
        ):
            issues.append(
                f"{tag}: none of the basis records allows action type {action.type.value}"
            )
        if action.type is ActionType.issue_refund and not unknown:
            issues += [f"{tag}: {m}" for m in _refund_issues(action, evidence)]
        if action.type is ActionType.request_more_info and not (clarification or "").strip():
            issues.append(
                f"{tag}: fill draft_clarification_question (a question for the CS officer to ask)"
            )
    return issues


def validate_draft(draft: ReportDraft, evidence: Evidence, feedback_text: str) -> list[str]:
    """All grounding problems of a draft ([] = accepted).

    Cited ids must come from this run's tool results, customer_context must match the record
    (or stay empty when not found), then check_summary and validate_actions.
    """
    issues: list[str] = []
    bad_gl = [
        guideline_id
        for guideline_id in draft.cited_guidelines
        if evidence.guideline(guideline_id) is None
    ]
    bad_pol = [
        policy_id for policy_id in draft.cited_policies if evidence.policy(policy_id) is None
    ]
    if bad_gl or bad_pol:
        issues.append(f"cited ids not present in tool results of this run: {bad_gl + bad_pol}")
    if not (draft.cited_guidelines or draft.cited_policies) and (
        evidence.guidelines or evidence.policies
    ):
        issues.append("cite at least one guideline or policy that you used")
    customer_context = draft.customer_context
    if evidence.customer_found:
        record = evidence.customer
        if (
            not customer_context.record_found
            or (
                customer_context.customer_id
                and customer_context.customer_id != record["customer_id"]
            )
            or (customer_context.tier and customer_context.tier != record["tier"])
        ):
            issues.append(
                f"customer_context must match the record: customer_id={evidence.customer['customer_id']}, "
                f"tier={evidence.customer['tier']}, record_found=true"
            )
    elif customer_context.record_found or customer_context.customer_id or customer_context.tier:
        issues.append(
            "customer was not found: use record_found=false and leave customer_id and tier empty"
        )
    issues += check_summary(draft.summary, feedback_text, evidence)
    issues += validate_actions(
        draft.suggested_actions, evidence, draft.draft_clarification_question
    )
    return issues

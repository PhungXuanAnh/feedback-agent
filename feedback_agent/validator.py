"""Grounding validator (code, not LLM): a report may only rely on records the tools returned in
THIS run. It checks that claims have a basis; it cannot check that a conclusion is right, which is
what the human officer is for."""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from .evidence import Evidence
from .schemas import ActionType, ReportDraft, SuggestedAction
from .tools import tier_rank

_MONEY = re.compile(r"(?:\$\s?(\d[\d,]*(?:\.\d+)?))|(?:(\d[\d,]*(?:\.\d+)?)\s?(?:USD|usd|dollars))")
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _norm(n: str) -> str:
    n = n.replace(",", "")
    return n[:-2] if n.endswith(".0") else (n.rstrip("0").rstrip(".") if "." in n else n)


# Money figures the CUSTOMER wrote. Heuristic by design (no language understanding): a number counts as
# money when it carries a currency marker ($N, N USD) or follows a charge/payment word. A number followed by a
# time unit ("37 days", "37.5 days", "1,250 days"), part of a date ("2026-09-10") or glued to an id ("INV-5005")
# never counts. The number must be matched whole: the lookahead forbids stopping before ".5" or ",250", so
# backtracking cannot turn "37.5 days" into the money figure 37.
_TIME_UNIT = r"(?:seconds?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|yrs?|%)"
_CHARGE_WORD = r"(?:charged|billed|paid|pay|refund(?:ed)?|costs?|fee|amount|price|total|owed?)"
_AFTER_CHARGE_WORD = re.compile(
    rf"\b{_CHARGE_WORD}\b[^\d$\n]{{0,25}}?(?<![A-Za-z0-9-])(\d[\d,]*(?:\.\d+)?)(?![\d-]|[.,]\d)(?!\s*{_TIME_UNIT}\b)", re.I)


def feedback_money(text: str) -> set[str]:
    marked = {_norm(m.group(1) or m.group(2)) for m in _MONEY.finditer(text)}
    return marked | {_norm(m.group(1)) for m in _AFTER_CHARGE_WORD.finditer(text)}


def check_summary(summary: str, feedback_text: str, evidence: Evidence) -> list[str]:
    """Money figures ($N or N USD) in the summary need a money source: a money figure in the feedback (see
    feedback_money), an invoice amount of this customer, or a policy limit. Digits in ids, dates and
    durations are not money sources. It does not check what the sentence CLAIMS about the money."""
    known = feedback_money(feedback_text) | evidence.money_values()
    issues = []
    for m in _MONEY.finditer(summary):
        if _norm(m.group(1) or m.group(2)) not in known:
            issues.append(f"summary mentions amount {m.group(0).strip()!r} that is not a money amount in the feedback, "
                          "an invoice of this customer or a policy limit")
    source = feedback_text + " " + evidence.numbers_text()
    for d in _ISO_DATE.findall(summary):
        if d not in source:
            issues.append(f"summary mentions date {d} that is in neither the feedback nor a tool result")
    return issues


def _refund_issues(a: SuggestedAction, ev: Evidence) -> list[str]:
    """issue_refund is the only money action: every condition is checked against run evidence."""
    pols = [ev.policy(b) for b in a.basis if ev.policy(b) and ActionType.issue_refund.value in ev.policy(b)["allowed_actions"]]
    if not pols:
        return ["issue_refund needs a POLICY in basis whose allowed_actions include issue_refund (a guideline is not enough)"]
    if not ev.customer_found:
        return ["issue_refund needs a verified customer record (lookup_customer returned not_found)"]
    p = a.params
    inv = next((i for i in ev.customer["recent_invoices"] if i["invoice_id"] == p.invoice_id), None)
    if inv is None:
        return [f"issue_refund invoice {p.invoice_id!r} is not an invoice of this customer in the lookup_customer result"]
    if p.amount is None or p.amount <= 0 or p.amount > inv["amount"]:
        return [f"issue_refund amount must be > 0 and <= invoice amount {inv['amount']}"]
    if inv["status"] != "paid":
        return [f"issue_refund invoice {inv['invoice_id']} is not paid (status={inv['status']})"]
    age = (ev.as_of - date.fromisoformat(inv["date"])).days
    dup_ids = {i for g in ev.customer["duplicate_charge_candidates"] for i in g["invoice_ids"]}
    per_policy: list[str] = []
    for pol in pols:  # valid if at least one cited refund policy has all its conditions satisfied
        bad = []
        if pol.get("max_amount") is not None and p.amount > pol["max_amount"]:
            bad.append(f"amount {p.amount} exceeds max_amount {pol['max_amount']}")
        if pol.get("window_days") is not None and not 0 <= age <= pol["window_days"]:
            bad.append(f"invoice is {age} days old, window is {pol['window_days']} days")
        if pol.get("requires_tier") and tier_rank(ev.customer["tier"]) < tier_rank(pol["requires_tier"]):
            bad.append(f"tier {ev.customer['tier']} is below required tier {pol['requires_tier']}")
        if pol.get("requires_duplicate_candidate") and inv["invoice_id"] not in dup_ids:
            bad.append("invoice is not one of the duplicate_charge_candidates")
        if not bad:
            return []
        per_policy.append(f"{pol['policy_id']}: " + "; ".join(bad))
    return ["issue_refund conditions not met -> " + " | ".join(per_policy)
            + " (if you cannot show them, propose request_more_info or escalate_to_human instead)"]


def validate_actions(actions: list[SuggestedAction], ev: Evidence, clarification: Optional[str]) -> list[str]:
    issues: list[str] = []
    n_refunds = sum(1 for a in actions if a.type is ActionType.issue_refund)
    if n_refunds > 1:
        # Policy limits such as "up to N USD per request" apply to the whole report, not to each action, and
        # two refunds of one invoice would pay it twice. Keep it simple: one refund per report, the officer
        # handles any further invoices (same rule for the machine's draft and for an override).
        issues.append(f"{n_refunds} issue_refund actions in one report; propose at most ONE refund per report "
                      "(mention other invoices in the summary and use escalate_to_human or request_more_info)")
    for n, a in enumerate(actions, 1):
        tag = f"action {n} ({a.type.value})"
        if not a.basis:
            issues.append(f"{tag}: basis is empty; cite guideline/policy ids from tool results")
            continue
        unknown = [b for b in a.basis if ev.record_for(b) is None]
        if unknown:
            issues.append(f"{tag}: basis ids not found in this run's tool results: {unknown}")
        if not any((r := ev.record_for(b)) and a.type.value in r["allowed_actions"] for b in a.basis):
            issues.append(f"{tag}: none of the basis records allows action type {a.type.value}")
        if a.type is ActionType.issue_refund and not unknown:
            issues += [f"{tag}: {m}" for m in _refund_issues(a, ev)]
        if a.type is ActionType.request_more_info and not (clarification or "").strip():
            issues.append(f"{tag}: fill draft_clarification_question (a question for the CS officer to ask)")
    return issues


def validate_draft(draft: ReportDraft, ev: Evidence, feedback_text: str) -> list[str]:
    issues: list[str] = []
    bad_gl = [g for g in draft.cited_guidelines if ev.guideline(g) is None]
    bad_pol = [p for p in draft.cited_policies if ev.policy(p) is None]
    if bad_gl or bad_pol:
        issues.append(f"cited ids not present in tool results of this run: {bad_gl + bad_pol}")
    if not (draft.cited_guidelines or draft.cited_policies) and (ev.guidelines or ev.policies):
        issues.append("cite at least one guideline or policy that you used")
    c = draft.customer_context
    if ev.customer_found:
        if not c.record_found or (c.customer_id and c.customer_id != ev.customer["customer_id"]) \
                or (c.tier and c.tier != ev.customer["tier"]):
            issues.append(f"customer_context must match the record: customer_id={ev.customer['customer_id']}, "
                          f"tier={ev.customer['tier']}, record_found=true")
    elif c.record_found or c.customer_id or c.tier:
        issues.append("customer was not found: use record_found=false and leave customer_id and tier empty")
    issues += check_summary(draft.summary, feedback_text, ev)
    issues += validate_actions(draft.suggested_actions, ev, draft.draft_clarification_question)
    return issues

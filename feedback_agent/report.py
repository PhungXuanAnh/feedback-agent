"""Final report assembly. Category, urgency, alternatives, flags and confidence are filled by
CODE; the LLM only contributes summary, citations, actions and the clarification draft."""
from __future__ import annotations

import uuid
from typing import Optional

from .evidence import SYSTEM_GUIDELINE, Evidence
from .schemas import (ActionParams, ActionType, Category, Classification, CustomerContext, Feedback, Report,
                      ReportDraft, SuggestedAction, Urgency, max_urgency)
from .validator import check_summary, validate_draft

_NO_POLICY_OK = {Category.praise, Category.feature_request}


def compute_confidence(cls: Classification, flags: set[str]) -> tuple[float, str]:
    """Heuristic, NOT a calibrated probability: start from the classifier's confidence and subtract
    for every weakness of the evidence. Weak runs are forced to `low` regardless of the score."""
    score = cls.confidence
    penalties = {"unverified_customer": 0.15, "no_applicable_policy": 0.10, "step_limit_reached": 0.15,
                 "injection_suspected": 0.10, "injection_in_tool_result": 0.10, "degraded": 0.30,
                 "grounding_failed": 0.30}
    score -= sum(p for f, p in penalties.items() if f in flags)
    if any(f.startswith("incomplete_context") for f in flags):
        score -= 0.25
    score = round(max(0.0, min(1.0, score)), 2)
    weak = {"degraded", "grounding_failed", "step_limit_reached", "needs_human_triage"} & flags or any(f.startswith("incomplete_context") for f in flags)
    level = "low" if weak else "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
    if level == "high" and {"unverified_customer", "injection_suspected", "injection_in_tool_result"} & flags:
        level = "medium"  # an unverified sender or adversarial text never earns "high"
    return score, level


def _customer_context(ev: Evidence, unavailable: bool) -> CustomerContext:
    if ev.customer_found:
        c = ev.customer
        return CustomerContext(record_found=True, customer_id=c["customer_id"], tier=c["tier"],
                               tenure_months=c["tenure_months"], open_tickets=len(c["open_tickets"]))
    note = ("customer source unavailable" if unavailable else
            "No customer record for this sender: unverified customer, tier unknown")
    return CustomerContext(record_found=False, note=note)


def _excerpt(text: str, n: int = 200) -> str:
    """Whitespace-normalised, cut at a word boundary so a figure or date is never split in half."""
    t = " ".join(text.split())
    if len(t) <= n:
        return t
    cut = t[:n]
    if t[n] != " " and " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "..."


def template_draft(fb: Feedback, cls: Classification, ev: Evidence, reason: str,
                   failed_sources: dict[str, str]) -> ReportDraft:
    """Report built by code from validated data only (no LLM). Only escalate_to_human and
    request_more_info are proposed, both based on the system guideline GL-TRIAGE."""
    from .schemas import DraftCustomerContext
    cust = ("customer record found, tier " + ev.customer["tier"] if ev.customer_found else
            "customer source unavailable" if "customer" in failed_sources else "customer not found (unverified)")
    parts = [f"Automatic fallback report ({reason}); no LLM analysis. Classified as {cls.category.value} "
             f"(urgency {cls.urgency.value}, sentiment {cls.sentiment.value}). Customer says: "
             f"\"{_excerpt(fb.text)}\". Context: {cust}."]
    if ev.guidelines:
        parts.append("Guideline retrieved: " + ", ".join(sorted(ev.guidelines)) + ".")
    if ev.policies:
        parts.append("Policies retrieved, not assessed: " + ", ".join(sorted(ev.policies)) + ".")
    if failed_sources:
        parts.append("Unavailable sources: " + ", ".join(sorted(failed_sources)) + ".")
    actions = [SuggestedAction(type=ActionType.escalate_to_human, params=ActionParams(target="cs_officer"),
                               basis=[SYSTEM_GUIDELINE], rationale="Human review needed: " + reason)]
    question = None
    if not failed_sources and (cls.needs_human_triage or not ev.customer_found):
        question = ("Could you describe what you need help with, the affected product area and, for billing "
                    "questions, the invoice number? Please also confirm the account email.")
        actions.append(SuggestedAction(type=ActionType.request_more_info, basis=[SYSTEM_GUIDELINE],
                                       rationale="Details are missing for a confident decision."))
    summary = " ".join(parts)[:900]
    if check_summary(summary, fb.text, ev):  # e.g. the 900-char cut or a lone huge token split a figure
        summary = (f"Automatic fallback report ({reason}); no LLM analysis. Classified as {cls.category.value} "
                   f"(urgency {cls.urgency.value}, sentiment {cls.sentiment.value}). Feedback excerpt omitted. "
                   f"Context: {cust}.")[:900]
    return ReportDraft(
        summary=summary, customer_context=DraftCustomerContext(
            record_found=ev.customer_found, customer_id=ev.customer["customer_id"] if ev.customer_found else None,
            tier=ev.customer["tier"] if ev.customer_found else None),
        cited_guidelines=sorted(ev.guidelines) + [SYSTEM_GUIDELINE], cited_policies=sorted(ev.policies),
        suggested_actions=actions, draft_clarification_question=question)


def build_report(*, fb: Feedback, cls: Classification, ev: Evidence, flags: set[str],
                 draft: Optional[ReportDraft], generated_by: str, provider: str, model: str, trace_id: str,
                 prompt_versions: dict[str, str], counters: dict, template_reason: str = "",
                 failed_sources: Optional[dict[str, str]] = None) -> Report:
    failed_sources = failed_sources or {}
    flags = set(flags)
    if draft is None:
        flags.add("degraded")
        draft = template_draft(fb, cls, ev, template_reason, failed_sources)
        problems = validate_draft(draft, ev, fb.text)
        assert not problems, f"degraded template must pass the validator: {problems}"  # a bug if it fires
    urgency: Urgency = cls.urgency
    if cls.category is Category.service_outage and ev.customer_found and ev.customer["tier"] == "enterprise":
        urgency = max_urgency(urgency, Urgency.high)
        if urgency != cls.urgency:
            flags.add("urgency_floor:enterprise_outage")
    if "customer" in ev.consulted and not ev.customer_found:
        flags.add("unverified_customer")
    if "policies" in ev.consulted and not ev.policies:
        flags.add("no_applicable_policy")
    if failed_sources:
        flags.add("incomplete_context:" + ",".join(sorted(failed_sources)))
    if cls.needs_human_triage:
        flags.add("needs_human_triage")
    for f, on in (("urgency_uncertain", cls.urgency_uncertain), ("truncated_input", fb.truncated),
                  ("classified_by_keyword_rules", cls.classified_by == "keyword_rules")):
        if on:
            flags.add(f)
    triage = (cls.needs_human_triage or "degraded" in flags or "grounding_failed" in flags
              or bool(failed_sources) or ("no_applicable_policy" in flags and cls.category not in _NO_POLICY_OK))
    if triage:
        flags.add("needs_human_triage")
    score, level = compute_confidence(cls, flags)  # triage => level low, whatever the classifier claimed
    return Report(
        report_id="rep_" + uuid.uuid4().hex[:10], feedback_id=fb.feedback_id or "", trace_id=trace_id,
        summary=draft.summary, category=cls.category, urgency=urgency, sentiment=cls.sentiment,
        alternatives=cls.alternatives, secondary_categories=draft.secondary_categories,
        customer_context=_customer_context(ev, "customer" in failed_sources),
        cited_policies=draft.cited_policies, cited_guidelines=draft.cited_guidelines,
        suggested_actions=draft.suggested_actions, draft_clarification_question=draft.draft_clarification_question,
        flags=sorted(flags), needs_human_triage=triage, confidence_score=score, confidence_level=level,
        prompt_versions=prompt_versions, report_generated_by=generated_by, provider=provider, model=model,
        as_of=ev.as_of.isoformat(), counters=counters)

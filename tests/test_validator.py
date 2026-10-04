import pytest

from feedback_agent.evidence import SYSTEM_GUIDELINE, Evidence
from feedback_agent.schemas import ReportDraft, SuggestedAction
from feedback_agent.tools import run_tool
from feedback_agent.validator import validate_actions, validate_draft
from conftest import AS_OF, DUP_TEXT


@pytest.fixture()
def ev(make_ctx, kb):
    e, ctx = Evidence(as_of=AS_OF), make_ctx()
    for name, args in [("get_cs_guidelines", {"category": "billing_issue"}),
                       ("lookup_customer", {"email": ctx.metadata_email}),
                       ("search_policies", {"query": "duplicate charge refund invoice", "category": "billing_issue"}),
                       ("search_policies", {"query": "sla breach refund", "category": "billing_issue"})]:
        e.record(run_tool(ctx, name, args))
    e.system_evidence[SYSTEM_GUIDELINE] = kb.guideline_by_id(SYSTEM_GUIDELINE)
    return e


def act(type_, basis, **params):
    return SuggestedAction(type=type_, basis=basis, params=params)


@pytest.mark.parametrize("action,expect", [
    (act("issue_refund", ["POL-DUP-01", "GL-BILL-01"], invoice_id="INV-5005", amount=1200), None),
    (act("issue_refund", ["GL-BILL-01"], invoice_id="INV-5005", amount=1200), "needs a POLICY"),
    (act("issue_refund", ["POL-DUP-01"], invoice_id="INV-5005", amount=1300), "<= invoice amount"),
    (act("issue_refund", ["POL-DUP-01"], invoice_id="INV-5001", amount=100), "not an invoice of this customer"),
    (act("issue_refund", ["POL-REF-02"], invoice_id="INV-5006", amount=1200), "window is 30 days"),  # 43 days old
    (act("issue_refund", ["POL-DUP-01"], invoice_id="INV-5006", amount=1200), "duplicate_charge_candidates"),
    (act("issue_refund", ["POL-REF-03"], invoice_id="INV-5005", amount=1200), "below required tier"),
    (act("issue_refund", ["POL-REF-01"], invoice_id="INV-5005", amount=1200), "not found in this run"),  # expired
    (act("escalate_to_engineering", ["GL-BILL-01"]), "allows action type"),
    (act("escalate_to_engineering", [SYSTEM_GUIDELINE]), "allows action type"),
    (act("escalate_to_human", [SYSTEM_GUIDELINE]), None),                    # system evidence supports triage only
])
def test_every_action_needs_an_allowing_basis_and_refunds_are_checked(ev, action, expect):
    issues = validate_actions([action], ev, clarification="q")
    assert (not issues) if expect is None else any(expect in i for i in issues)


def test_draft_level_checks(ev):
    base = dict(summary="Customer says they were charged twice; the record shows INV-5004 and INV-5005.",
                customer_context={"record_found": True, "customer_id": "C-1004", "tier": "pro"},
                cited_policies=["POL-DUP-01"], cited_guidelines=["GL-BILL-01"],
                suggested_actions=[act("escalate_to_human", ["GL-BILL-01"])])
    ok = ReportDraft(**base)
    assert validate_draft(ok, ev, DUP_TEXT) == []
    bad = ReportDraft(**{**base, "summary": "We should refund $9,999 right away.", "cited_policies": ["POL-NOPE"],
                         "customer_context": {"record_found": True, "customer_id": "C-1004", "tier": "enterprise"},
                         "suggested_actions": [act("request_more_info", ["GL-BILL-01"])]})
    text = " | ".join(validate_draft(bad, ev, DUP_TEXT))
    for part in ("$9,999", "POL-NOPE", "must match the record", "draft_clarification_question"):
        assert part in text


def test_money_guards_duplicate_refunds_nan_and_id_digits(ev):
    refund = act("issue_refund", ["POL-DUP-01", "GL-BILL-01"], invoice_id="INV-5005", amount=1200)
    twice = validate_actions([refund, refund.model_copy(deep=True)], ev, None)
    assert any("ONE refund per report" in i for i in twice)              # 2 x 1200 on one 1200 invoice
    other = act("issue_refund", ["POL-REF-02"], invoice_id="INV-5004", amount=1200)
    two_invoices = validate_actions([refund, other], ev, None)           # 2400 against a "2000 per request" policy
    assert any("ONE refund per report" in i for i in two_invoices)
    assert validate_actions([refund], ev, None) == []                    # a single valid refund is untouched
    with pytest.raises(ValueError):
        SuggestedAction(type="issue_refund", basis=["POL-DUP-01"], params={"invoice_id": "INV-5005", "amount": "NaN"})
    ok = dict(customer_context={"record_found": True, "customer_id": "C-1004", "tier": "pro"}, cited_policies=["POL-DUP-01"],
              cited_guidelines=["GL-BILL-01"], suggested_actions=[act("escalate_to_human", ["GL-BILL-01"])])
    # "1004" only exists as digits of the id C-1004; 1,200.00 is a real invoice amount
    assert any("$1004" in i for i in validate_draft(ReportDraft(summary="The record confirms a $1004 refund.", **ok), ev, DUP_TEXT))
    assert validate_draft(ReportDraft(summary="Invoice INV-5005 was $1,200.00 and 1200 USD was paid.", **ok), ev, DUP_TEXT) == []


def test_feedback_money_needs_a_money_context(ev):
    from feedback_agent.validator import check_summary
    ok_ev = Evidence(as_of=AS_OF)
    for text in ("I paid $777.00.", "I paid 777 USD.", "I was charged 777.", "We were billed 1,250.50 twice.",
                 "I was charged 1,250 yesterday."):
        assert check_summary("Customer says they paid $%s." % (("1250.50" if "1,250.50" in text else "1250" if "1,250" in text else "777")), text, ok_ev) == [], text
    for text, fake in (("I have waited 37 days for support.", "$37"), ("This started on 2026-09-10.", "$2026"),
                       ("I was charged 37 days ago and it is invoice 777.", "$37"), ("See INV-5005 for details.", "$5005"),
                       ("Please look at invoice 777.", "$777"),
                       ("I was charged 37.5 days ago.", "$37"), ("I was charged 1,250 days ago.", "$1"),
                       ("I paid it 0.5 hours ago.", "$0"), ("I was billed 12.25 weeks back.", "$12")):
        assert check_summary(f"The customer requests a {fake} refund.", text, ok_ev), text

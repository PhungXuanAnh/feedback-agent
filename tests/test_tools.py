from datetime import date

import pytest

from feedback_agent.tools import KnowledgeBase, ToolStatus, run_tool


def test_guidelines_found_and_not_found(make_ctx, kb):
    ctx = make_ctx()
    r = run_tool(ctx, "get_cs_guidelines", {"category": "billing_issue"})
    assert r.status is ToolStatus.found and r.data["guideline_id"] == "GL-BILL-01"
    ctx.kb = KnowledgeBase([g for g in kb.guidelines if g["category"] != "praise"], kb.policies)
    assert run_tool(ctx, "get_cs_guidelines", {"category": "praise"}).status is ToolStatus.not_found
    bad = run_tool(ctx, "get_cs_guidelines", {"category": "nonsense"})
    assert bad.status is ToolStatus.error and bad.error == "invalid_arguments"  # error != not_found


def test_customer_found_notfound_and_scope_lock(make_ctx):
    r = run_tool(make_ctx(), "lookup_customer", {"email": "billing@orchid-robotics.example"})
    assert r.status is ToolStatus.found and r.data["tier"] == "pro"
    assert r.data["duplicate_charge_candidates"][0]["invoice_ids"] == ["INV-5004", "INV-5005"]
    ghost = run_tool(make_ctx("ghost@unknown-corp.example"), "lookup_customer",
                     {"email": "ghost@unknown-corp.example"})
    assert ghost.status is ToolStatus.not_found and ghost.consulted
    other = run_tool(make_ctx(), "lookup_customer", {"email": "ops@northwind-analytics.example"})
    assert other.status is ToolStatus.error and other.error == "out_of_scope"
    assert "scope_violation" in other.flags and not other.consulted
    assert run_tool(make_ctx(), "lookup_customer", {"customer_id": "C-1001"}).error == "out_of_scope"


def test_policy_search_is_bounded_and_not_found(make_ctx):
    r = run_tool(make_ctx(), "search_policies", {"query": "duplicate charge refund", "category": "billing_issue"})
    ids = [p["policy_id"] for p in r.data["policies"]]
    assert r.status is ToolStatus.found and len(ids) <= 3 and ids[0] == "POL-DUP-01"
    assert run_tool(make_ctx(), "search_policies", {"query": "zzzz qqqq"}).status is ToolStatus.not_found


@pytest.mark.parametrize("as_of,present,absent", [
    (date(2024, 12, 31), [], ["POL-REF-01", "POL-REF-02"]),           # before first effective date
    (date(2025, 1, 1), ["POL-REF-01"], ["POL-REF-02"]),                # start day is inclusive
    (date(2026, 6, 30), ["POL-REF-01"], ["POL-REF-02"]),               # last day of v1
    (date(2026, 7, 1), ["POL-REF-02"], ["POL-REF-01"]),                # expiry day excluded, v2 starts
    (date(2026, 3, 1), ["POL-SLA-01"], ["POL-SLA-00"]),                # superseded although it never expires
])
def test_policy_versions(kb, as_of, present, absent):
    ids = {p["policy_id"] for p in kb.active_policies(as_of)}
    assert set(present) <= ids and not (set(absent) & ids)
    hits = {p["policy_id"] for p in kb.search_policies("refund invoice sla response", "billing_issue", 3, as_of)}
    assert not (set(absent) & hits)  # an expired/superseded policy is never retrievable, hence never citable

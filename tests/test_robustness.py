"""The three failure situations the brief requires: ambiguous input, missing record, LLM/tool failure."""
from conftest import C, DUP_EMAIL, G, P, calls, classify_step, fb, submit
from feedback_agent.tools import FaultInjector
from feedback_agent.trace import load_trace

GHOST = "ghost@unknown-corp.example"


def test_ambiguous_input_goes_to_triage_with_alternatives_and_clarification(run):
    alts = [{"category": "billing_issue", "confidence": 0.3}, {"category": "bug_report", "confidence": 0.25}]
    gl = ("get_cs_guidelines", {"category": "unclear"})
    ask = {"type": "request_more_info", "basis": ["GL-TRIAGE"], "rationale": "Cannot tell what is wrong."}
    rep, _ = run([classify_step("unclear", 0.35, "low", alts), calls(gl, C, ("search_policies", {"query": "help"})),
                  submit([ask], cited_pol=(), cited_gl=("GL-TRIAGE",),
                         summary="Customer says something is wrong with their account but gives no details.",
                         draft_clarification_question="Which feature is affected, and since when?")],
                 fb("it does not work, please fix", DUP_EMAIL))
    assert rep.needs_human_triage and rep.category.value == "unclear" and rep.urgency.value == "medium"
    assert [a.category.value for a in rep.alternatives] == ["billing_issue", "bug_report"]
    assert rep.suggested_actions[0].type.value == "request_more_info" and rep.draft_clarification_question
    assert rep.confidence_level == "low"


def test_missing_customer_record_is_not_found_not_error(run):
    ghost_c = ("lookup_customer", {"email": GHOST})
    rep, _ = run([classify_step(), calls(G, ghost_c, P),
                  submit([{"type": "request_more_info", "basis": ["GL-BILL-01"], "rationale": "Unverified sender."}],
                         found=False, cited_pol=(),
                         summary="Customer says they were charged twice; no customer record exists for this sender.",
                         draft_clarification_question="Please confirm the account email and the invoice number.")],
                 fb(email=GHOST))
    assert rep.customer_context.record_found is False and rep.customer_context.tier is None
    assert "unverified_customer" in rep.flags and rep.report_generated_by == "scripted"
    assert rep.confidence_score < 0.9 and rep.confidence_level == "medium"  # unverified sender: never "high"


def test_llm_failures_retry_then_degrade_and_keep_good_results(run, tmp_path):
    t = {"error": "transient"}
    # one transient failure: retried inside the same turn, no degradation, no extra turn used
    rep, prov = run([classify_step(), t, calls(G, C, P), submit()])
    assert rep.report_generated_by == "scripted" and rep.counters["llm_turns"] == 2 and prov.calls == 4
    # LLM dies after the tools ran: retry fails too -> template that reuses the 3 successful results
    rep, _ = run([classify_step(), calls(G, C, P), t, t])
    assert rep.report_generated_by == "degraded_template" and {"degraded", "llm_unavailable"} <= set(rep.flags)
    assert rep.counters["degraded_calls"] == 0 and rep.cited_guidelines[0] == "GL-BILL-01"
    assert rep.confidence_level == "low" and rep.needs_human_triage
    kinds = [e["event"] for e in load_trace(rep.trace_id, str(tmp_path / "traces"))]
    assert kinds.count("llm_error") == 2 and "degraded_path" in kinds
    # LLM down from the start: keyword classification, the 3 sources fetched without the LLM
    rep, _ = run([t, t])
    assert rep.report_generated_by == "degraded_template" and rep.counters["degraded_calls"] == 3
    assert "classified_by_keyword_rules" in rep.flags and rep.category.value == "billing_issue"
    assert rep.customer_context.tier == "pro"


def test_tool_errors_retry_once_then_incomplete_context_never_not_found(run):
    # transient source error: retried once automatically, the run is normal
    rep, _ = run([classify_step(), calls(G, C, P), submit()], faults=FaultInjector({"search_policies": 1}))
    assert rep.report_generated_by == "scripted" and "incomplete_context:policies" not in rep.flags
    # source stays broken: error != not_found, so the evidence gate is NOT satisfied -> incomplete_context
    rep, prov = run([classify_step(), calls(G, C, P)], faults=FaultInjector({"search_policies": -1}))
    assert rep.report_generated_by == "degraded_template" and prov.calls == 2  # finite: no further LLM turns
    assert "incomplete_context:policies" in rep.flags and "no_applicable_policy" not in rep.flags
    assert [a.type.value for a in rep.suggested_actions] == ["escalate_to_human"]
    assert rep.suggested_actions[0].basis == ["GL-TRIAGE"] and rep.confidence_level == "low"


def test_broken_report_store_raises_instead_of_claiming_queued(run, store, monkeypatch):
    import pytest
    from feedback_agent.store import StorageError

    def boom(*a, **k):
        raise StorageError("disk full")
    monkeypatch.setattr(store, "save_report", boom)
    with pytest.raises(StorageError):
        run([classify_step(), calls(G, C, P), submit()])


def test_fallback_survives_figure_at_excerpt_boundary_and_bad_tool_args(run):
    from feedback_agent.llm.base import LLMTurn, ToolCall
    # a $ figure straddling the 200-char excerpt limit used to crash the template (assert on the validator)
    text = "billing " + "x" * 188 + "$1234567 extra details"
    rep, _ = run([classify_step(), {"error": "transient"}, {"error": "transient"}], feedback=fb(text=text))
    assert rep.report_generated_by == "degraded_template" and "$123..." not in rep.summary
    # one unbroken 300-char token: the excerpt cannot cut at a space, the summary check still protects the report
    rep, _ = run([classify_step(), {"error": "transient"}, {"error": "transient"}], feedback=fb(text="$1234567" + "y" * 300))
    assert rep.report_generated_by == "degraded_template"
    # a provider turn whose tool args are not an object degrades instead of crashing the loop
    bad = LLMTurn(tool_calls=[ToolCall("c1", "lookup_customer", ["not-an-object"])])
    rep, _ = run([classify_step(), bad])
    assert rep.report_generated_by == "degraded_template" and rep.needs_human_triage

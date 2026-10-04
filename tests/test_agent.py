import pytest

from conftest import C, G, P, calls, classify_step, submit
from feedback_agent.trace import load_trace


def test_happy_path_is_grounded_and_traced(run, tmp_path):
    rep, prov = run([classify_step(), calls(G, C, P), submit()])
    assert rep.report_generated_by == "scripted" and rep.status == "pending_review"
    assert rep.cited_policies == ["POL-DUP-01"] and rep.suggested_actions[0].type.value == "issue_refund"
    assert rep.customer_context.tier == "pro" and rep.confidence_level == "high" and rep.flags == []
    assert prov.calls == 3 and rep.counters["llm_turns"] == 2 and rep.counters["tool_calls"] == 3
    events = [e["event"] for e in load_trace(rep.trace_id, str(tmp_path / "traces"))]
    assert events.count("tool_call") == 3 and "classification_completed" in events and "report_generated" in events


def test_evidence_gate_rejects_early_submit_then_accepts(run):
    rep, prov = run([classify_step(), calls(G, C), submit(), calls(P), submit()])
    assert rep.report_generated_by == "scripted" and rep.counters["llm_turns"] == 4
    gate = [m for m in prov.seen[3] if m.role == "tool"][-1].tool_responses[0].response
    assert gate["error"] == "missing_sources" and gate["missing_sources"] == ["policies"]


def test_caps_are_enforced(run):
    # max_tool_calls=2: the repeated guideline call is a free cache hit, the 3rd distinct call is refused
    # (even when retried), the early submit hits the gate, and the reserved last turn lets the system
    # run the missing lookup itself (outside the cap) before the forced submit.
    rep, prov = run([classify_step(), calls(G, G, C, P), calls(P), submit(), submit()],
                    max_tool_calls=2, max_llm_turns=4)
    first = [r.response for m in prov.seen[2] if m.role == "tool" for r in m.tool_responses]
    assert first[3]["error"] == "tool_budget_exhausted"
    assert (rep.counters["tool_calls"], rep.counters["cache_hits"], rep.counters["forced_by_system"]) == (2, 1, 1)
    assert prov.calls == 5 and "step_limit_reached" in rep.flags and rep.confidence_level == "low"
    # minimal turns: one free turn, then the reserved one with system lookups for the other two sources
    rep, prov = run([classify_step(), calls(G), submit()], max_llm_turns=2)
    assert rep.counters["forced_by_system"] == 2 and prov.calls == 3 and rep.report_generated_by == "scripted"


def test_grounding_fix_once_then_template_without_exceeding_cap(run, tmp_path):
    bad = submit(cited_pol=("POL-FAKE-99",))
    rep, _ = run([classify_step(), calls(G, C, P), bad, submit()])  # fixed on the second try
    assert rep.report_generated_by == "scripted" and rep.counters["llm_turns"] == 3
    # still wrong at the reserved last turn: no more LLM calls, report rebuilt from validated data
    rep, prov = run([classify_step(), calls(G, C, P), bad, bad], max_llm_turns=3)
    assert prov.calls == 4 and rep.report_generated_by == "degraded_template" and "grounding_failed" in rep.flags
    assert {a.type.value for a in rep.suggested_actions} <= {"escalate_to_human", "request_more_info"}
    assert "POL-FAKE-99" not in rep.cited_policies and rep.confidence_level == "low" and rep.needs_human_triage
    trace = load_trace(rep.trace_id, str(tmp_path / "traces"))
    assert any(e.get("rejected_draft") for e in trace if e["event"] == "grounding_validation")

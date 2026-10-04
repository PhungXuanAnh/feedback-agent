import json

from conftest import C, G, P, calls, classify_step, submit
from feedback_agent.config import Settings
from feedback_agent.evaluation import render_markdown, run_eval
from feedback_agent.llm import ScriptedProvider

CASE = {"id": "dup", "text": "I was charged twice on 2026-09-10, 1200 USD each time. Please refund the duplicate payment.",
        "customer_email": "billing@orchid-robotics.example", "received_at": "2026-09-22T09:00:00Z",
        "label": {"category": "billing_issue", "min_urgency": "low", "needs_triage": False, "customer_found": True,
                  "expected_guidelines": ["GL-BILL-01"], "expected_policies": ["POL-DUP-01"],
                  "forbidden_actions": ["escalate_to_engineering"]}}


def test_eval_scores_runs_and_lists_misses(tmp_path):
    wrong = {**CASE, "id": "mislabelled", "label": {**CASE["label"], "category": "praise"}}  # a deliberate miss
    degraded = {**CASE, "id": "llm-down"}
    scripts = {"dup": [classify_step(), calls(G, C, P), submit()], "mislabelled": [classify_step(), calls(G, C, P), submit()],
               "llm-down": [classify_step(), calls(G, C, P), {"error": "transient"}, {"error": "transient"}]}
    s = Settings(llm_provider="scripted", llm_retry_backoff_s=0, data_dir=str(__import__("conftest").DATA))
    res = run_eval(s, [CASE, wrong, degraded], lambda c: ScriptedProvider(scripts[c["id"]]), workers=1,
                   trace_root=str(tmp_path / "t"))
    sm = res["configs"][0]["summary"]
    assert sm["runs"] == 3 and sm["errors"] == 0 and sm["category_accuracy"] == "2/3"
    assert sm["completed_normally"] == "2/3" and sm["degraded_runs"] == 1
    assert sm["retrieval_hit_rate"] == {"guidelines": "3/3", "policies": "3/3"}
    assert sm["fabricated_refs_in_final_reports"] == 0 and sm["runs_within_caps"] == "3/3" and sm["traces_complete"] == "3/3"
    assert [c["case"] for c in sm["case_results"] if c["why"]] == ["mislabelled", "llm-down"]
    assert "mislabelled" in render_markdown(res) and json.dumps(res)


def test_eval_counts_crashed_runs_as_failures_and_never_reports_all_clear(tmp_path):
    def factory(case):
        if case["id"] == "boom":
            raise RuntimeError("controlled failure")
        return ScriptedProvider([classify_step(), calls(G, C, P), submit()])
    s = Settings(llm_provider="scripted", llm_retry_backoff_s=0, data_dir=str(__import__("conftest").DATA))
    res = run_eval(s, [CASE, {**CASE, "id": "boom"}], factory, workers=1, trace_root=str(tmp_path / "t"))
    sm = res["configs"][0]["summary"]
    assert (sm["runs"], sm["errors"], sm["completed_normally"], sm["category_accuracy"]) == (2, 1, "1/2", "1/2")
    assert sm["error_cases"][0]["case"] == "boom"
    md = render_markdown(res)
    assert "crashed" in md and "No case missed" not in md

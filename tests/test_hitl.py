import json

import pytest
from fastapi.testclient import TestClient

from conftest import C, G, P, calls, classify_step, fb, submit
from feedback_agent.api import create_app
from feedback_agent.config import Settings
from feedback_agent.hitl import (Overrides, OverrideRejected, ReportNotFound, ReviewConflict, ReviewRequest,
                                 ReviewService)
from feedback_agent.schemas import SuggestedAction
from feedback_agent.pipeline import FeedbackPipeline
from feedback_agent.llm import ScriptedProvider


def req(decision, **kw):
    return ReviewRequest(decision=decision, actor="officer.lee", **kw)


@pytest.fixture()
def svc(store, tmp_path):
    return ReviewService(store, Settings(trace_dir=str(tmp_path / "traces")))


def new_report(run):
    return run([classify_step(), calls(G, C, P), submit()])[0].report_id


def test_review_flow_approve_idempotent_reject_terminal_override(run, svc, store):
    # approve -> executed once; repeating the same decision replays the old result, never executes again
    rid = new_report(run)
    assert store.get_report_row(rid)["status"] == "pending_review"
    first = svc.review(rid, req("approve"))
    assert first["status"] == "executed" and first["execution"][0].startswith("POST /billing/refunds")
    again = svc.review(rid, req("approve"))
    assert again["idempotent"] is True and len(svc.executor.calls) == 1
    with pytest.raises(ReviewConflict):
        svc.review(rid, req("reject"))                   # conflicting decision -> 409
    assert len(store.reviews_for(rid)) == 1              # audit untouched, decision kept after executed

    # reject is terminal: nothing is ever executed afterwards
    rid2 = new_report(run)
    assert svc.review(rid2, req("reject", note="not a duplicate"))["status"] == "rejected"
    with pytest.raises(ReviewConflict):
        svc.review(rid2, req("approve"))
    assert len(svc.executor.calls) == 1

    # override: validated by the same grounding rules, machine suggestion preserved with before/after
    rid3 = new_report(run)
    bad = Overrides(suggested_actions=[SuggestedAction(type="issue_refund", basis=["POL-DUP-01"],
                                                      params={"invoice_id": "INV-5001", "amount": 5})])
    with pytest.raises(OverrideRejected):
        svc.review(rid3, req("override", overrides=bad))
    assert store.get_report_row(rid3)["status"] == "pending_review"   # a failed override changes nothing
    good = Overrides(urgency="high", suggested_actions=[
        SuggestedAction(type="escalate_to_human", basis=["GL-BILL-01"], params={"target": "billing_lead"})])
    out = svc.review(rid3, req("override", overrides=good))
    detail = svc.get(rid3)
    assert out["status"] == "executed" and svc.executor.calls[-1].startswith("POST /escalations/human")
    audit = detail["reviews"][0]
    assert audit["overrides"]["before"]["suggested_actions"][0]["type"] == "issue_refund"
    assert audit["overrides"]["after"]["urgency"] == "high" and audit["machine_urgency"] == "medium"
    assert detail["report"]["suggested_actions"][0]["type"] == "issue_refund"  # machine proposal kept
    with pytest.raises(ReportNotFound):
        svc.review("rep_missing", req("approve"))


def test_api_queue_and_review(store, kb, tmp_path):
    s = Settings(llm_provider="scripted", trace_dir=str(tmp_path / "traces"), llm_retry_backoff_s=0)
    pipe = FeedbackPipeline(s, store, kb, ScriptedProvider([classify_step(), calls(G, C, P), submit()]))
    client = TestClient(create_app(s, pipe))
    body = {"text": "I was charged twice on 2026-09-10, 1200 USD each time.", "customer_email": "billing@orchid-robotics.example",
            "received_at": "2026-09-22T09:00:00Z"}
    first = client.post("/feedback", json=body).json()
    rid = first["report_id"]
    assert first["as_of"] == "2026-09-22"      # the client-supplied timestamp fixes the policy date
    assert [r["report_id"] for r in client.get("/reports", params={"status": "pending_review"}).json()] == [rid]
    assert client.post("/feedback", json={"text": "x", "customer_email": "nope"}).status_code == 422
    r = {"decision": "approve", "actor": "officer.lee"}
    assert client.post(f"/reports/{rid}/review", json=r).json()["status"] == "executed"
    assert client.post(f"/reports/{rid}/review", json={**r, "decision": "reject"}).status_code == 409
    assert client.get("/reports/rep_nope").status_code == 404
    assert client.get("/reports", params={"status": "pending_review"}).json() == []


def test_api_trace_endpoint_takes_ids_only(store, kb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "canary.json").write_text('{"private": 1}\n')   # a file outside trace_dir must stay unreachable
    s = Settings(llm_provider="scripted", trace_dir=str(tmp_path / "traces"), llm_retry_backoff_s=0)
    pipe = FeedbackPipeline(s, store, kb, ScriptedProvider([classify_step(), calls(G, C, P), submit()]))
    client = TestClient(create_app(s, pipe))
    tid = client.post("/feedback", json={"text": "I was charged twice on 2026-09-10, 1200 USD.",
                                         "customer_email": "billing@orchid-robotics.example"}).json()["trace_id"]
    assert client.get(f"/traces/{tid}").status_code == 200
    assert client.get("/traces/canary.json").status_code == 404 and client.get("/traces/..%2Fcanary").status_code == 404

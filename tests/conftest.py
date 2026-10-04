from datetime import date
from pathlib import Path

import pytest

from feedback_agent.store import Store
from feedback_agent.tools import KnowledgeBase, ToolContext

DATA = Path(__file__).resolve().parent.parent / "data"
AS_OF = date(2026, 9, 22)  # fixed "now" for every test


@pytest.fixture()
def store(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.seed_from_csv(DATA / "seed")
    return s


@pytest.fixture()
def kb():
    return KnowledgeBase.load(DATA)


@pytest.fixture()
def make_ctx(store, kb):
    def _make(email="billing@orchid-robotics.example", customer_id=None, as_of=AS_OF, **kw):
        return ToolContext(store=store, kb=kb, metadata_email=email, metadata_customer_id=customer_id,
                           as_of=as_of, **kw)
    return _make


# ---------------------------------------------------------------- agent helpers
from datetime import datetime, timezone  # noqa: E402

from feedback_agent.config import Settings  # noqa: E402
from feedback_agent.llm import ScriptedProvider  # noqa: E402
from feedback_agent.pipeline import FeedbackPipeline  # noqa: E402
from feedback_agent.schemas import Feedback  # noqa: E402

DUP_TEXT = "I was charged twice on 2026-09-10, 1200 USD each time. Please refund the duplicate payment."
DUP_EMAIL = "billing@orchid-robotics.example"


def fb(text=DUP_TEXT, email=DUP_EMAIL, **kw):
    return Feedback(text=text, customer_email=email, received_at=datetime(2026, 9, 22, 9, tzinfo=timezone.utc), **kw)


def classify_step(cat="billing_issue", conf=0.9, urgency="medium", alts=None, sentiment="negative"):
    return {"tool_calls": [{"name": "classify_feedback", "args": {
        "category": cat, "alternatives": alts or [], "sentiment": sentiment, "urgency": urgency,
        "confidence": conf, "rationale": "test"}}]}


def calls(*names_args):
    return {"tool_calls": [{"name": n, "args": a} for n, a in names_args]}


G = ("get_cs_guidelines", {"category": "billing_issue"})
C = ("lookup_customer", {"email": DUP_EMAIL})
P = ("search_policies", {"query": "duplicate charge refund", "category": "billing_issue"})


def submit(actions=None, cited_pol=("POL-DUP-01",), cited_gl=("GL-BILL-01",), found=True, summary=None, **kw):
    refund = {"type": "issue_refund", "params": {"invoice_id": "INV-5005", "amount": 1200},
              "basis": ["POL-DUP-01", "GL-BILL-01"], "rationale": "duplicate"}
    args = {"summary": summary or "Customer says they were charged twice; the record shows a duplicate invoice.",
            "customer_context": {"record_found": found, "customer_id": "C-1004" if found else None,
                                 "tier": "pro" if found else None},
            "cited_policies": list(cited_pol), "cited_guidelines": list(cited_gl),
            "suggested_actions": actions or [refund], **kw}
    return {"tool_calls": [{"name": "submit_report", "args": args}]}


@pytest.fixture()
def run(store, kb, tmp_path):
    def _run(script, feedback=None, faults=None, **settings):
        s = Settings(llm_provider="scripted", trace_dir=str(tmp_path / "traces"), llm_retry_backoff_s=0, **settings)
        prov = script if hasattr(script, "generate") else ScriptedProvider(script)
        rep = FeedbackPipeline(s, store, kb, prov, faults).process(feedback or fb())
        return rep, prov
    return _run

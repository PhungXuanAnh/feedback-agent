from fastapi.testclient import TestClient

from conftest import C, G, P, calls, classify_step, submit
from feedback_agent.api import create_app
from feedback_agent.config import Settings
from feedback_agent.llm import ScriptedProvider
from feedback_agent.llm.base import LLMTurn, ToolCall
from feedback_agent.pipeline import FeedbackPipeline

BODY = {"text": "I was charged twice on 2026-09-10, 1200 USD each time.", "customer_email": "billing@orchid-robotics.example"}


def _client(store, kb, tmp_path, script, **limits):
    s = Settings(llm_provider="scripted", trace_dir=str(tmp_path / "t"), llm_retry_backoff_s=0, **limits)
    return TestClient(create_app(s, FeedbackPipeline(s, store, kb, ScriptedProvider(script))))


def test_token_rate_limit_and_daily_caps(store, kb, tmp_path):
    one = [classify_step(), calls(G, C, P), submit()]
    # token + rate limit (2 per minute): health and docs stay open, the rest needs the header
    c = _client(store, kb, tmp_path, one * 2, api_token="secret", rate_limit_per_min=2)
    h = {"X-API-Token": "secret"}
    assert c.get("/health").status_code == 200 and c.get("/docs").status_code == 200
    assert c.get("/reports").status_code == 401 and c.get("/reports", headers={"X-API-Token": "bad"}).status_code == 401
    assert c.post("/feedback", json=BODY).status_code == 401
    assert [c.post("/feedback", json=BODY, headers=h).status_code for _ in range(2)] == [201, 201]
    limited = c.post("/feedback", json=BODY, headers=h)
    assert limited.status_code == 429 and int(limited.headers["Retry-After"]) >= 1 and c.get("/reports", headers=h).status_code == 200
    # daily request cap counts persisted reports (2 exist now), so a restart does not reset it
    c = _client(store, kb, tmp_path, one, daily_request_limit=2)
    assert c.post("/feedback", json=BODY).status_code == 429
    # daily token budget: each scripted call reports 1100 tokens, the 2 stored reports used far more than 1000
    used = [LLMTurn(tool_calls=[ToolCall(f"x{i}", t["name"], t["args"]) for i, t in enumerate(step["tool_calls"])],
                    usage={"prompt_tokens": 600, "output_tokens": 500}) for step in one]
    c = _client(store, kb, tmp_path, used, daily_token_limit=1000)
    assert store.usage_since("2000-01-01")[0] == 2 and c.post("/feedback", json=BODY).status_code == 201
    assert c.post("/feedback", json=BODY).status_code == 429       # that single run used 3300 tokens > 1000

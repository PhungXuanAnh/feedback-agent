import pytest
from conftest import C, G, P, calls, classify_step, submit
from feedback_agent.llm.base import LLMTurn, ToolCall
from feedback_agent.trace import load_trace, render_trace


def _turn(step, usage):
    return LLMTurn(tool_calls=[ToolCall(f"x{i}", c["name"], c["args"]) for i, c in enumerate(step["tool_calls"])],
                   usage=usage, raw={"parts": [{"thought": True, "text": "PRIVATE-REASONING"}]})


def test_trace_has_usage_cost_and_never_the_models_reasoning(run, tmp_path):
    u = {"prompt_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}
    steps = [_turn(s, u) for s in (classify_step(), calls(G, C, P), submit())]
    rep, _ = run(steps, price_in=1.0, price_out=2.0)   # USD per million tokens
    assert rep.counters["prompt_tokens"] == 3000 and rep.counters["output_tokens"] == 1500
    assert rep.counters["cost_usd"] == 0.006  # (3000*1 + 1500*2) / 1e6
    text = (tmp_path / "traces" / f"{rep.trace_id}.jsonl").read_text()
    assert "PRIVATE-REASONING" not in text
    events = load_trace(rep.trace_id, str(tmp_path / "traces"))
    assert all(e["trace_id"] == rep.trace_id for e in events)
    assert "tool_result" in render_trace(events) and "classification_completed" in render_trace(events)
    # no usage reported (scripted) or no prices configured: cost is "unknown", never 0
    rep, _ = run([classify_step(), calls(G, C, P), submit()])
    assert rep.counters["cost_usd"] == "unknown" and rep.counters["prompt_tokens"] is None


def test_failed_attempt_usage_is_counted_and_missing_usage_is_flagged(tmp_path):
    import httpx
    from feedback_agent.config import Settings
    from feedback_agent.llm import GeminiProvider
    from feedback_agent.llm.base import Message, ToolChoice
    from feedback_agent.llmcaller import LLMCaller
    from feedback_agent.trace import Tracer
    replies = [{"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}],   # failed attempt, still billed
                "usageMetadata": {"promptTokenCount": 100, "thoughtsTokenCount": 50, "totalTokenCount": 150}},
               {"candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20, "totalTokenCount": 120}}]
    prov = GeminiProvider("k", "m", client=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json=replies.pop(0)))))
    s = Settings(trace_dir=str(tmp_path), price_in=1, price_out=1, llm_retry_backoff_s=0)
    tr = Tracer("tr_000000000001", s)
    LLMCaller(prov, s, tr).call("t", "v1", [Message("user", "x")], [], ToolChoice())
    t = tr.totals()
    assert (t["prompt_tokens"], t["output_tokens"], t["cost_usd"]) == (200, 70, 0.00027)
    assert (t["llm_calls"], t["provider_calls"], t["usage_complete"]) == (1, 2, True)
    tr2 = Tracer("tr_000000000002", s)       # an answered call that reports no usage: totals are a lower bound
    tr2.record_llm_call({"prompt_tokens": 10, "output_tokens": 5}); tr2.record_llm_call(None)
    assert tr2.totals()["usage_complete"] is False


@pytest.mark.parametrize("failure,complete", [("non_json_200", False), ("read_timeout", False),
                                              ("connect_error", True), ("http_503", True)])
def test_usage_complete_distinguishes_unknown_from_known_free_attempts(failure, complete, tmp_path):
    import httpx
    from feedback_agent.config import Settings
    from feedback_agent.llm import GeminiProvider
    from feedback_agent.llm.base import Message, ToolChoice
    from feedback_agent.llmcaller import LLMCaller
    from feedback_agent.trace import Tracer
    calls_seen = []

    def handle(request):
        calls_seen.append(1)
        if len(calls_seen) == 1:
            if failure == "read_timeout":
                raise httpx.ReadTimeout("t", request=request)
            if failure == "connect_error":
                raise httpx.ConnectError("c", request=request)
            return httpx.Response(503) if failure == "http_503" else httpx.Response(200, text="not JSON")
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                                         "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20}})
    prov = GeminiProvider("k", "m", client=httpx.Client(transport=httpx.MockTransport(handle)))
    s = Settings(trace_dir=str(tmp_path), llm_retry_backoff_s=0, price_in=1, price_out=1)
    tr = Tracer("tr_000000000003", s)
    LLMCaller(prov, s, tr).call("t", "v1", [Message("user", "x")], [], ToolChoice())
    t = tr.totals()
    assert t["provider_calls"] == 2 and t["usage_complete"] is complete
    assert (t["prompt_tokens"], t["output_tokens"]) == (100, 20)    # the known part is always reported

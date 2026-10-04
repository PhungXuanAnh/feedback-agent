import json

import httpx
import pytest

from feedback_agent.config import Settings
from feedback_agent.llm import ConfigError, GeminiProvider, ScriptedProvider, get_provider
from feedback_agent.llm.base import LLMError, Message, ToolChoice, ToolResponse, ToolSpec
from feedback_agent.llm.gemini import to_gemini_schema
from feedback_agent.llm.scripted import ScriptError
from feedback_agent.tools import tool_specs

# shape of a real generateContent response: parallel calls, ids, thought signature on the first call
GEMINI_REPLY = {
    "candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"id": "fc1", "name": "get_cs_guidelines", "args": {"category": "billing_issue"}},
         "thoughtSignature": "SIG-1"},
        {"functionCall": {"id": "fc2", "name": "lookup_customer", "args": {"email": "a@b.example"}}}]},
        "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 50,
                      "totalTokenCount": 170},
}


def test_factory_picks_class_by_name_and_rejects_unknown():
    s = Settings(llm_provider="scripted")
    assert isinstance(get_provider(s, script=[]), ScriptedProvider)
    assert isinstance(get_provider(Settings(llm_provider="gemini", llm_api_key="k", llm_model="m")), GeminiProvider)
    with pytest.raises(ConfigError, match="Unknown LLM_PROVIDER"):
        get_provider(Settings(llm_provider="nope"))
    with pytest.raises(ConfigError, match="API_KEY"):
        get_provider(Settings(llm_provider="gemini", llm_model="m"))


def test_gemini_http_roundtrip_replays_raw_and_answers_every_call():
    sent = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append((req, json.loads(req.content)))
        return httpx.Response(200, json=GEMINI_REPLY)

    p = GeminiProvider("secret-key", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    msgs = [Message("system", "sys"), Message("user", "hi")]
    turn = p.generate(msgs, tool_specs(), ToolChoice("any", ["lookup_customer"]), 256)
    assert [c.id for c in turn.tool_calls] == ["fc1", "fc2"] and turn.usage["output_tokens"] == 70
    req, body = sent[0]
    assert req.headers["x-goog-api-key"] == "secret-key" and "secret-key" not in str(req.url)
    assert body["toolConfig"]["functionCallingConfig"] == {"mode": "ANY", "allowedFunctionNames": ["lookup_customer"]}
    assert "$ref" not in json.dumps(body["tools"])  # enum schema inlined, not dropped
    # second request: model turn replayed verbatim (thought signature kept), ONE user message, 2 responses
    msgs += [turn.as_message(), Message("tool", tool_responses=[
        ToolResponse("fc1", "get_cs_guidelines", {"status": "found"}),
        ToolResponse("fc2", "lookup_customer", {"status": "not_found"})])]
    p.generate(msgs, tool_specs(), ToolChoice("auto"), 256)
    contents = sent[1][1]["contents"]
    assert contents[1]["parts"][0]["thoughtSignature"] == "SIG-1"
    resp_parts = contents[2]["parts"]
    assert contents[2]["role"] == "user" and [x["functionResponse"]["id"] for x in resp_parts] == ["fc1", "fc2"]


def test_gemini_error_classification():
    def run(status, body=None):
        p = GeminiProvider("k", "m", client=httpx.Client(transport=httpx.MockTransport(
            lambda r: httpx.Response(status, json=body or {}))))
        with pytest.raises(LLMError) as e:
            p.generate([Message("user", "x")], [], ToolChoice())
        return e.value.transient
    assert run(503) and run(429) and not run(403, {"error": {"message": "bad key"}})
    assert run(200, {"candidates": []})  # empty answer is retryable once
    assert run(200, {"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}]})  # sampling noise: retry


def test_scripted_provider_rejects_bad_conversation_and_tool_choice():
    spec = [ToolSpec("t", "d", {"type": "object", "properties": {}})]
    p = ScriptedProvider([{"tool_calls": [{"name": "t"}]}, {"tool_calls": [{"name": "t"}]}])
    turn = p.generate([Message("user", "x")], spec, ToolChoice())
    with pytest.raises(ScriptError, match="answered"):  # a call left without its response
        p.generate([Message("user", "x"), turn.as_message(), Message("user", "again")], spec, ToolChoice())
    assert to_gemini_schema({"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}
                            )["properties"]["a"] == {"type": "string"}


@pytest.mark.parametrize("payload", [[], "x", {"candidates": "no"},
                                     {"candidates": [{"content": {"parts": [{"functionCall": {"args": {}}}]}}]},
                                     {"candidates": [{"content": {"parts": [{"functionCall": {"name": "t", "args": [1]}}]}}]}])
def test_malformed_gemini_payloads_become_llm_errors_and_are_retryable(payload):
    with pytest.raises(LLMError) as e:
        GeminiProvider.parse(payload)
    p = GeminiProvider("k", "m", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="not json"))))
    with pytest.raises(LLMError, match="not JSON"):
        p.generate([Message("user", "x")], [], ToolChoice())
    assert e.value.transient


@pytest.mark.parametrize("body,expect", [({"error": {"message": "Invalid request"}}, "Invalid request"),
                                         ([], "HTTP 400"), ({"error": None}, "HTTP 400"), ("plain text", "HTTP 400")])
def test_http_400_with_any_body_shape_is_an_llm_error(body, expect):
    p = GeminiProvider("k", "m", client=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(400, json=body) if not isinstance(body, str) else httpx.Response(400, text=body))))
    with pytest.raises(LLMError, match=expect) as e:
        p.generate([Message("user", "x")], [], ToolChoice())
    assert not e.value.transient and not e.value.usage_unknown     # a rejected call: known free, not retried

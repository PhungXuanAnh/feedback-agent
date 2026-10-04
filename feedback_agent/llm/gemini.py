"""Gemini adapter over the REST generateContent API (httpx, no SDK).

Idea borrowed from a personal project (logilens): Pydantic JSON-Schema -> Gemini function
declaration. Rewritten here for a multi-turn tool loop: model turns are replayed verbatim
(`raw`, which carries thought signatures) and every functionCall gets a functionResponse."""
from __future__ import annotations

import copy
from typing import Any, Optional

import httpx

from .base import LLMError, LLMTurn, Message, ToolCall, ToolChoice, ToolSpec

_ALLOWED_FORMATS = {"date", "date-time", "enum"}
_DROP = {"$defs", "$ref", "title", "default", "additionalProperties", "examples"}


def to_gemini_schema(schema: dict[str, Any], defs: Optional[dict] = None) -> dict[str, Any]:
    """Reduce a Pydantic JSON-Schema to the OpenAPI subset Gemini accepts ($ref inlined)."""
    if not isinstance(schema, dict):
        return schema
    defs = defs if defs is not None else schema.get("$defs", {})
    if "$ref" in schema:
        target = defs.get(schema["$ref"].split("/")[-1], {})
        return to_gemini_schema({**copy.deepcopy(target), **{k: v for k, v in schema.items() if k != "$ref"}}, defs)
    if "allOf" in schema and len(schema["allOf"]) == 1:
        rest = {k: v for k, v in schema.items() if k != "allOf"}
        return to_gemini_schema({**schema["allOf"][0], **rest}, defs)
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if not (isinstance(s, dict) and s.get("type") == "null")]
        rest = {k: v for k, v in schema.items() if k != "anyOf"}
        if len(non_null) == 1:
            return to_gemini_schema({**rest, **non_null[0]}, defs)
        return {"type": "string"}
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k in _DROP or (k == "format" and v not in _ALLOWED_FORMATS):
            continue
        if k == "properties":
            out[k] = {pk: to_gemini_schema(pv, defs) for pk, pv in v.items()}
        elif k == "items":
            out[k] = to_gemini_schema(v, defs)
        else:
            out[k] = v
    if "type" not in out and "enum" not in out:
        out["type"] = "string"
    return out


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, base_url: str = "https://generativelanguage.googleapis.com",
                 temperature: Optional[float] = None, timeout: float = 60.0, thinking_level: str = "",
                 client: Optional[httpx.Client] = None):
        self._key = api_key
        self.model = model
        self._base = base_url.rstrip("/")
        self._temperature = temperature
        self._thinking = thinking_level
        self._client = client or httpx.Client(timeout=timeout)

    # -------------------------------------------------------------- request
    def _contents(self, messages: list[Message]) -> tuple[Optional[dict], list[dict]]:
        system = "\n\n".join(m.text for m in messages if m.role == "system")
        contents: list[dict] = []
        api_ids: set[str] = set()
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "user":
                contents.append({"role": "user", "parts": [{"text": m.text}]})
            elif m.role == "model":
                if m.raw is not None:  # replay verbatim: keeps thought signatures intact
                    contents.append(m.raw)
                    api_ids |= {p["functionCall"]["id"] for p in m.raw.get("parts", [])
                                if "functionCall" in p and p["functionCall"].get("id")}
                else:
                    parts: list[dict] = [{"text": m.text}] if m.text else []
                    parts += [{"functionCall": {"name": c.name, "args": c.args}} for c in m.tool_calls]
                    contents.append({"role": "model", "parts": parts})
            elif m.role == "tool":  # all responses of one model turn go into ONE user message
                parts = []
                for r in m.tool_responses:
                    fr: dict[str, Any] = {"name": r.name, "response": r.response}
                    if r.call_id in api_ids:
                        fr["id"] = r.call_id
                    parts.append({"functionResponse": fr})
                if m.text:
                    parts.append({"text": m.text})
                contents.append({"role": "user", "parts": parts})
        return ({"parts": [{"text": system}]} if system else None), contents

    def build_body(self, messages: list[Message], tools: list[ToolSpec], tool_choice: ToolChoice,
                   max_output_tokens: int) -> dict[str, Any]:
        system, contents = self._contents(messages)
        body: dict[str, Any] = {"contents": contents,
                                "generationConfig": {"maxOutputTokens": max_output_tokens}}
        if self._temperature is not None:
            body["generationConfig"]["temperature"] = self._temperature
        if self._thinking:  # thinking tokens count against maxOutputTokens, so a lower level also avoids truncation
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self._thinking}
        if system:
            body["system_instruction"] = system
        if tools:
            body["tools"] = [{"functionDeclarations": [
                {"name": t.name, "description": t.description, "parameters": to_gemini_schema(t.parameters)}
                for t in tools]}]
            cfg: dict[str, Any] = {"mode": tool_choice.mode.upper()}
            if tool_choice.mode == "any" and tool_choice.allowed:
                cfg["allowedFunctionNames"] = tool_choice.allowed
            body["toolConfig"] = {"functionCallingConfig": cfg}
        return body

    # ------------------------------------------------------------- response
    def generate(self, messages: list[Message], tools: list[ToolSpec], tool_choice: ToolChoice,
                 max_output_tokens: int = 2048) -> LLMTurn:
        body = self.build_body(messages, tools, tool_choice, max_output_tokens)
        url = f"{self._base}/v1beta/models/{self.model}:generateContent"
        try:
            r = self._client.post(url, json=body, headers={"x-goog-api-key": self._key})
        except httpx.TransportError as e:  # includes timeouts
            # a failure while CONNECTING never reached the model (no tokens); a read timeout/error may have
            # happened after the model started working, so its usage is unknown
            connect_phase = isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
            raise LLMError(f"network error: {type(e).__name__}", transient=True, usage_unknown=not connect_phase) from e
        # HTTP error statuses: the provider rejected the call. Google documents that failed 400/500 requests
        # are not billed for tokens (they still count toward quota); other statuses are assumed the same.
        if r.status_code == 429 or r.status_code >= 500:
            raise LLMError(f"HTTP {r.status_code}", transient=True)
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code}: {self._error_message(r)}")
        try:
            payload = r.json()
        except ValueError as e:  # a proxy or gateway can answer 200 with a non-JSON body
            raise LLMError("response body is not JSON", transient=True, usage_unknown=True) from e
        return self.parse(payload)

    @staticmethod
    def _error_message(r: httpx.Response) -> str:
        """Best-effort text of an HTTP error body; any unexpected shape yields an empty string, never an exception."""
        try:
            err = r.json().get("error")
            msg = err.get("message") if isinstance(err, dict) else None
            return msg[:200] if isinstance(msg, str) else ""
        except (ValueError, AttributeError):
            return ""

    @staticmethod
    def _usage(payload: Any) -> Optional[dict]:
        u = payload.get("usageMetadata") if isinstance(payload, dict) else None
        if not isinstance(u, dict) or not u:
            return None
        return {"prompt_tokens": u.get("promptTokenCount"),
                "output_tokens": (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0),
                "total_tokens": u.get("totalTokenCount")}

    @classmethod
    def parse(cls, payload: Any) -> LLMTurn:
        """generateContent response -> LLMTurn. Anything malformed becomes an LLMError that still carries
        the usage the provider reported, because a failed attempt is billed too."""
        usage = cls._usage(payload)
        try:
            turn = cls._parse(payload)
        except LLMError as e:
            e.usage, e.usage_unknown = usage, usage is None
            raise
        turn.usage = usage
        return turn

    @staticmethod
    def _parse(payload: Any) -> LLMTurn:
        if not isinstance(payload, dict):
            raise LLMError("unexpected response shape (not an object)", transient=True)
        cands = payload.get("candidates") or []
        if not isinstance(cands, list) or not cands or not isinstance(cands[0], dict):
            pf = payload.get("promptFeedback")
            block = pf.get("blockReason") if isinstance(pf, dict) else None
            raise LLMError(f"no usable candidates (blockReason={block})", transient=block is None)
        cand = cands[0]
        content = cand.get("content") if isinstance(cand.get("content"), dict) else {}
        raw_parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        parts = [p for p in raw_parts if isinstance(p, dict)]
        calls = []
        for i, p in enumerate(x for x in parts if "functionCall" in x):
            fc = p["functionCall"]
            if not isinstance(fc, dict) or not isinstance(fc.get("name"), str) or not fc["name"]:
                raise LLMError("malformed functionCall (missing name)", transient=True)
            args = {} if fc.get("args") is None else fc["args"]
            if not isinstance(args, dict):
                raise LLMError("malformed functionCall (args are not an object)", transient=True)
            calls.append(ToolCall(id=fc.get("id") or f"call_{i}", name=fc["name"], args=args))
        text = "".join(p.get("text", "") for p in parts if isinstance(p.get("text"), str) and not p.get("thought")).strip()
        if not calls and not text:
            # MALFORMED_FUNCTION_CALL is sampling noise on a complex schema: worth one retry
            reason = cand.get("finishReason")
            raise LLMError(f"empty response (finishReason={reason}; {str(cand.get('finishMessage', ''))[:160]})",
                           transient=reason in (None, "STOP", "OTHER", "MALFORMED_FUNCTION_CALL"))
        if not calls and cand.get("finishReason") == "MAX_TOKENS":
            raise LLMError("response truncated (MAX_TOKENS)")
        raw = {"role": "model", "parts": parts}
        return LLMTurn(text=text, tool_calls=calls, raw=raw)

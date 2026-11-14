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
_GEMINI_MODES = {"auto": "AUTO", "required": "ANY", "none": "NONE"}  # functionCallingConfig.mode


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
        non_null = [option for option in schema["anyOf"]
                    if not (isinstance(option, dict) and option.get("type") == "null")]
        rest = {k: v for k, v in schema.items() if k != "anyOf"}
        if len(non_null) == 1:
            return to_gemini_schema({**rest, **non_null[0]}, defs)
        return {"type": "string"}
    reduced: dict[str, Any] = {}
    for k, v in schema.items():
        if k in _DROP or (k == "format" and v not in _ALLOWED_FORMATS):
            continue
        if k == "properties":
            reduced[k] = {property_name: to_gemini_schema(property_schema, defs) for property_name,
                          property_schema in v.items()}
        elif k == "items":
            reduced[k] = to_gemini_schema(v, defs)
        else:
            reduced[k] = v
    if "type" not in reduced and "enum" not in reduced:
        reduced["type"] = "string"
    return reduced


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, base_url: str = "https://generativelanguage.googleapis.com",
                 temperature: Optional[float] = None, timeout: float = 60.0, thinking_level: str = "",
                 client: Optional[httpx.Client] = None):
        self._api_key = api_key
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._thinking = thinking_level
        self._client = client or httpx.Client(timeout=timeout)

    # -------------------------------------------------------------- request
    def _contents(self, messages: list[Message]) -> tuple[Optional[dict], list[dict]]:
        system = "\n\n".join(message.text for message in messages if message.role == "system")
        contents: list[dict] = []
        api_ids: set[str] = set()
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "user":
                contents.append({"role": "user", "parts": [{"text": message.text}]})
            elif message.role == "model":
                if message.raw is not None:  # replay verbatim: keeps thought signatures intact
                    contents.append(message.raw)
                    api_ids |= {part["functionCall"]["id"] for part in message.raw.get("parts", [])
                                if "functionCall" in part and part["functionCall"].get("id")}
                else:
                    parts: list[dict] = [{"text": message.text}] if message.text else []
                    parts += [{"functionCall": {"name": call.name, "args": call.args}} for call in message.tool_calls]
                    contents.append({"role": "model", "parts": parts})
            elif message.role == "tool":  # all responses of one model turn go into ONE user message
                parts = []
                for response in message.tool_responses:
                    function_response: dict[str, Any] = {"name": response.name, "response": response.response}
                    if response.call_id in api_ids:
                        function_response["id"] = response.call_id
                    parts.append({"functionResponse": function_response})
                if message.text:
                    parts.append({"text": message.text})
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
                {"name": tool.name, "description": tool.description, "parameters": to_gemini_schema(tool.parameters)}
                for tool in tools]}]
            calling_config: dict[str, Any] = {"mode": _GEMINI_MODES[tool_choice.mode]}
            if tool_choice.mode == "required" and tool_choice.allowed:
                calling_config["allowedFunctionNames"] = tool_choice.allowed
            body["toolConfig"] = {"functionCallingConfig": calling_config}
        return body

    # ------------------------------------------------------------- response
    def generate(self, messages: list[Message], tools: list[ToolSpec], tool_choice: ToolChoice,
                 max_output_tokens: int = 2048) -> LLMTurn:
        body = self.build_body(messages, tools, tool_choice, max_output_tokens)
        url = f"{self._base_url}/v1beta/models/{self.model}:generateContent"
        try:
            response = self._client.post(url, json=body, headers={"x-goog-api-key": self._api_key})
        except httpx.TransportError as e:  # includes timeouts
            # a failure while CONNECTING never reached the model (no tokens); a read timeout/error may have
            # happened after the model started working, so its usage is unknown
            connect_phase = isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
            raise LLMError(f"network error: {type(e).__name__}", transient=True, usage_unknown=not connect_phase) from e
        # HTTP error statuses: the provider rejected the call. Google documents that failed 400/500 requests
        # are not billed for tokens (they still count toward quota); other statuses are assumed the same.
        if response.status_code == 429 or response.status_code >= 500:
            raise LLMError(f"HTTP {response.status_code}", transient=True)
        if response.status_code >= 400:
            raise LLMError(f"HTTP {response.status_code}: {self._error_message(response)}")
        try:
            payload = response.json()
        except ValueError as e:  # a proxy or gateway can answer 200 with a non-JSON body
            raise LLMError("response body is not JSON", transient=True, usage_unknown=True) from e
        return self.parse(payload)

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        """Best-effort text of an HTTP error body; any unexpected shape yields an empty string, never an exception."""
        try:
            error = response.json().get("error")
            message = error.get("message") if isinstance(error, dict) else None
            return message[:200] if isinstance(message, str) else ""
        except (ValueError, AttributeError):
            return ""

    @staticmethod
    def _usage(payload: Any) -> Optional[dict]:
        usage = payload.get("usageMetadata") if isinstance(payload, dict) else None
        if not isinstance(usage, dict) or not usage:
            return None
        return {"prompt_tokens": usage.get("promptTokenCount"),
                "output_tokens": (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0),
                "total_tokens": usage.get("totalTokenCount")}

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
        candidates = payload.get("candidates") or []
        if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
            prompt_feedback = payload.get("promptFeedback")
            block = prompt_feedback.get("blockReason") if isinstance(prompt_feedback, dict) else None
            raise LLMError(f"no usable candidates (blockReason={block})", transient=block is None)
        candidate = candidates[0]
        content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
        raw_parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        parts = [part for part in raw_parts if isinstance(part, dict)]
        calls = []
        for i, part in enumerate(x for x in parts if "functionCall" in x):
            function_call = part["functionCall"]
            if (not isinstance(function_call, dict) or not isinstance(function_call.get("name"), str)
                    or not function_call["name"]):
                raise LLMError("malformed functionCall (missing name)", transient=True)
            args = {} if function_call.get("args") is None else function_call["args"]
            if not isinstance(args, dict):
                raise LLMError("malformed functionCall (args are not an object)", transient=True)
            calls.append(ToolCall(id=function_call.get("id") or f"call_{i}", name=function_call["name"], args=args))
        text = "".join(part.get("text", "") for part in parts
                       if isinstance(part.get("text"), str) and not part.get("thought")).strip()
        if not calls and not text:
            # MALFORMED_FUNCTION_CALL is sampling noise on a complex schema: worth one retry
            reason = candidate.get("finishReason")
            raise LLMError(f"empty response (finishReason={reason}; {str(candidate.get('finishMessage', ''))[:160]})",
                           transient=reason in (None, "STOP", "OTHER", "MALFORMED_FUNCTION_CALL"))
        if not calls and candidate.get("finishReason") == "MAX_TOKENS":
            raise LLMError("response truncated (MAX_TOKENS)")
        raw = {"role": "model", "parts": parts}
        return LLMTurn(text=text, tool_calls=calls, raw=raw)

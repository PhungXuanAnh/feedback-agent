"""Defences that keep untrusted text as data: customer feedback and tool results are escaped
and wrapped in tags before any LLM reads them, so they cannot close or fake a delimiter.
The other layers are schema-forced output, read-only tools, the customer scope lock, the
grounding validator and human review."""

from __future__ import annotations

import json
from typing import Any


def escape(text: str) -> str:
    """Neutralise tag characters so data can never close or fake a delimiter."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def wrap_feedback(text: str) -> str:
    """Escape the customer's text inside <customer_feedback> tags, so the LLM reads it as data."""
    return f"<customer_feedback>\n{escape(text)}\n</customer_feedback>"


def wrap_tool_result(tool: str, payload: dict[str, Any], origin: str = "llm") -> str:
    """Escape a tool result (JSON) inside <tool_result tool="..." origin="..."> tags."""
    body = escape(json.dumps(payload, ensure_ascii=False, default=str))
    return f'<tool_result tool="{tool}" origin="{origin}">\n{body}\n</tool_result>'

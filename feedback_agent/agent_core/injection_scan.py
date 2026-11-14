"""Step 2, injection scan: regexes FLAG known prompt-injection patterns; they never block.

Regexes are easy to evade and a real customer may write "please ignore my previous email", so a
match only adds a flag. The defences that stop injection are in guard.py and later steps.
"""

from __future__ import annotations

import re

PATTERNS = {
    "ignore_instructions": r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|all|any|your|the)\b.{0,30}\b(instruction|rule|prompt|guideline|polic)",
    "role_override": r"\b(you are now|act as|pretend (to be|you are)|new instructions?|system prompt|developer mode|jailbreak)\b",
    "reveal_prompt": r"\b(reveal|show|print|repeat|output)\b.{0,30}\b(system|hidden|initial)\b.{0,20}\b(prompt|instruction)",
    "forced_action": r"\b(mark|set|close|flag)\b.{0,30}\b(as )?(resolved|closed|approved|done)\b",
    "forced_refund": r"\b(approve|issue|process|grant)\b.{0,30}\b(full |a |the )?(refund|credit|payment)\b.{0,40}\b(without|immediately|now|regardless|no questions)\b",
    "fake_delimiter": r"</?\s*(customer_feedback|tool_result|system|assistant)\b",
    "note_to_ai": r"\b(note|message|instruction)s? (to|for) (the )?(support )?(assistant|ai|agent|bot|model)\b",
}
_COMPILED = {k: re.compile(v, re.I | re.S) for k, v in PATTERNS.items()}


def find_injection_patterns(text: str) -> list[str]:
    """Names of the injection patterns that match (empty = nothing suspicious)."""
    return [name for name, rx in _COMPILED.items() if rx.search(text)]

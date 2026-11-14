"""Step 2, injection scan: regexes FLAG known prompt-injection patterns; they never block.

Regexes are easy to evade, and a real customer may write "please ignore my previous
instructions about the invoice", so a match only adds a flag. The defences that stop injection
are in guard.py and later steps.
"""

from __future__ import annotations

import re
import unicodedata

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


# Characters a human reviewer cannot see: zero-width and bidi controls, and Unicode tag
# characters (U+E0000-E007F), which can spell out a whole hidden ASCII instruction. U+200D
# (zero-width joiner) is left out because emoji use it; subdivision flags (England, Scotland,
# Wales) use tag characters and are flagged.
_HIDDEN = re.compile(
    "[\u200b\u200c\u200e\u200f\u202a-\u202e\u2060-\u2064\ufeff\U000e0000-\U000e007f]"
)
_TAG_BASE = 0xE0000


def _screening_copy(text: str) -> str:
    """Copy of the text that the regexes read; the LLM still gets the original text.

    Tag characters are spelled out as the ASCII they encode, the result is NFKC-normalised
    (full-width letters become ASCII) and the remaining hidden characters are removed, so
    "ig<zero-width space>nore" reads as "ignore".
    """
    chars = []
    for char in text:
        if 0xE0020 <= ord(char) <= 0xE007E:
            char = chr(ord(char) - _TAG_BASE)
        chars.append(char)
    return _HIDDEN.sub("", unicodedata.normalize("NFKC", "".join(chars)))


def find_injection_patterns(text: str) -> list[str]:
    """Names of the injection patterns that match (empty = nothing suspicious).

    Hidden characters are reported as a pattern of their own, even when the rest is harmless.
    """
    screened = _screening_copy(text)
    names = [name for name, rx in _COMPILED.items() if rx.search(screened)]
    if _HIDDEN.search(text):
        names.append("hidden_characters")
    return names

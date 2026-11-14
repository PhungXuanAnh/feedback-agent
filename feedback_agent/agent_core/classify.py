"""Step 3: classification. One forced-schema LLM call + code-owned safety nets
(urgency floor rules, human-triage flag) + a keyword fallback when the LLM is down."""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from ..llm.base import Message, ToolChoice, ToolSpec
from ..llm.caller import LLMCaller, LLMUnavailable
from ..schemas import (
    Alternative,
    Category,
    Classification,
    Feedback,
    Sentiment,
    Urgency,
    max_urgency,
)
from .guard import wrap_feedback
from .prompts import CLASSIFY_PROMPT, load_prompt

TRIAGE_CONFIDENCE = 0.65  # below this the primary category is not trusted (conservative default)
TRIAGE_ALT_CONFIDENCE = 0.40  # a second reading this plausible means the message is ambiguous


class ClassifyArgs(BaseModel):
    category: Category
    alternatives: list[Alternative] = Field(
        default_factory=list,
        max_length=2,
        description="Up to 2 other plausible categories with confidence.",
    )
    sentiment: Sentiment
    urgency: Urgency
    confidence: float = Field(ge=0, le=1, description="Confidence in the primary category, 0-1.")
    rationale: str = Field(default="", max_length=300)


CLASSIFY_TOOL = ToolSpec(
    "classify_feedback",
    "Record the classification of the customer feedback.",
    ClassifyArgs.model_json_schema(),
)

# Floor rules: code sets a minimum urgency, the LLM can only raise it (under-rating an incident
# costs more than over-rating a small issue).
_SEVERE = re.compile(
    r"\b(breach(ed)?|data leak|leaked|lawsuit|legal action|attorney|lawyer|sue (you|us)|"
    r"ransomware|unauthori[sz]ed access|hacked|regulator)\b",
    re.I,
)


def apply_safety_rules(classification: Classification, text: str) -> Classification:
    """Code-owned safety rules on top of a classification; they can only RAISE urgency.

    security_concern or a severe word (breach, lawsuit, ...) -> at least high; unclear or
    confidence < 0.65 -> at least medium. needs_human_triage when unclear, low confidence, or an
    alternative reading has confidence >= 0.40. Example: billing/low + "we will sue you" -> high.
    """
    reasons: list[str] = []
    floor = Urgency.low
    if classification.category is Category.security_concern:
        floor = Urgency.high
        reasons.append("security_concern")
    if _SEVERE.search(text):
        floor = Urgency.high
        reasons.append("severe_term")
    if classification.category is Category.unclear or classification.confidence < TRIAGE_CONFIDENCE:
        classification.urgency_uncertain = True  # unknown urgency defaults to medium, never lower
        floor = max_urgency(floor, Urgency.medium)
        reasons.append("urgency_uncertain_default_medium")
    classification.urgency = max_urgency(classification.urgency, floor)
    classification.urgency_floor_reasons = reasons
    classification.needs_human_triage = (
        classification.category is Category.unclear
        or classification.confidence < TRIAGE_CONFIDENCE
        or any(
            alternative.confidence >= TRIAGE_ALT_CONFIDENCE
            for alternative in classification.alternatives
        )
    )
    return classification


_KEYWORDS = [
    (Category.security_concern, r"breach|leak|hacked|unauthori[sz]ed|vulnerab|credential"),
    (Category.data_privacy, r"gdpr|personal data|delete my data|data deletion|privacy"),
    (
        Category.service_outage,
        r"\bdown\b|outage|unavailable|not working at all|cannot access|can't access",
    ),
    (Category.billing_issue, r"invoice|charged|charge|refund|billing|payment"),
    (Category.churn_risk, r"cancel|switch(ing)? to|competitor|leaving|terminate our"),
    (Category.bug_report, r"bug|error|crash|fails?|broken|wrong"),
    (Category.feature_request, r"feature|would like|wish|please add|support for"),
    (Category.praise, r"thank|great|love|excellent|awesome"),
]
_ABUSIVE = re.compile(r"\b(idiot|stupid|useless|scam|garbage|incompetent|moron|damn)\b", re.I)


def classify_by_keywords(text: str) -> Classification:
    """Degraded path when the LLM is unavailable: coarse, never confident, always sent to triage.

    The first matching keyword rule wins (security first): "charged twice" -> billing_issue.
    """
    category = Category.unclear
    for candidate, rx in _KEYWORDS:
        if re.search(rx, text, re.I):
            category = candidate
            break
    sentiment = (
        Sentiment.abusive
        if _ABUSIVE.search(text)
        else Sentiment.positive if category is Category.praise else Sentiment.negative
    )
    classification = Classification(
        category=category,
        sentiment=sentiment,
        urgency=Urgency.medium,
        confidence=0.4,
        rationale="keyword rules (LLM unavailable)",
        classified_by="keyword_rules",
    )
    return apply_safety_rules(classification, text)


def classify_by_llm(llm_caller: LLMCaller, feedback: Feedback) -> Classification:
    """Classify with ONE forced classify_feedback call, then apply_safety_rules.

    Raises LLMUnavailable when the provider fails or the answer is missing or invalid; the
    pipeline then uses classify_by_keywords.
    """
    messages = [
        Message("system", load_prompt(CLASSIFY_PROMPT)),
        Message("user", f"{wrap_feedback(feedback.text)}\n\nChannel: {feedback.channel.value}"),
    ]
    llm_turn = llm_caller.call(
        "classify",
        CLASSIFY_PROMPT,
        messages,
        [CLASSIFY_TOOL],
        ToolChoice("required", ["classify_feedback"]),
    )
    call = None
    for tool_call in llm_turn.tool_calls:
        if tool_call.name == "classify_feedback":
            call = tool_call
            break
    if call is None:
        raise LLMUnavailable("classify_feedback was not called")
    try:
        args = ClassifyArgs.model_validate(call.args)
    except ValidationError as e:
        raise LLMUnavailable(f"invalid classification: {e.error_count()} schema errors") from e
    classification = Classification(**args.model_dump(), classified_by="llm")
    return apply_safety_rules(classification, feedback.text)

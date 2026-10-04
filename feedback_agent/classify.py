"""Step 3: classification. One forced-schema LLM call + code-owned safety nets
(urgency floor rules, human-triage flag) + a keyword fallback when the LLM is down."""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from .guard import wrap_feedback
from .llm.base import Message, ToolChoice, ToolSpec
from .llmcaller import LLMCaller, LLMUnavailable
from .prompts import CLASSIFY_PROMPT, load_prompt
from .schemas import Alternative, Category, Classification, Feedback, Sentiment, Urgency, max_urgency

TRIAGE_CONFIDENCE = 0.65      # below this the primary category is not trusted (conservative default)
TRIAGE_ALT_CONFIDENCE = 0.40  # a second reading this plausible means the message is ambiguous


class ClassifyArgs(BaseModel):
    category: Category
    alternatives: list[Alternative] = Field(default_factory=list, max_length=2,
                                            description="Up to 2 other plausible categories with confidence.")
    sentiment: Sentiment
    urgency: Urgency
    confidence: float = Field(ge=0, le=1, description="Confidence in the primary category, 0-1.")
    rationale: str = Field(default="", max_length=300)


CLASSIFY_TOOL = ToolSpec("classify_feedback", "Record the classification of the customer feedback.",
                         ClassifyArgs.model_json_schema())

# Floor rules: code sets a minimum urgency, the LLM can only raise it (under-rating an incident
# costs more than over-rating a small issue).
_SEVERE = re.compile(r"\b(breach(ed)?|data leak|leaked|lawsuit|legal action|attorney|lawyer|sue (you|us)|"
                     r"ransomware|unauthori[sz]ed access|hacked|regulator)\b", re.I)


def apply_floor(cls: Classification, text: str) -> Classification:
    reasons: list[str] = []
    floor = Urgency.low
    if cls.category is Category.security_concern:
        floor = Urgency.high
        reasons.append("security_concern")
    if _SEVERE.search(text):
        floor = Urgency.high
        reasons.append("severe_term")
    if cls.category is Category.unclear or cls.confidence < TRIAGE_CONFIDENCE:
        cls.urgency_uncertain = True  # unknown urgency defaults to medium, never lower
        floor = max_urgency(floor, Urgency.medium)
        reasons.append("urgency_uncertain_default_medium")
    cls.urgency = max_urgency(cls.urgency, floor)
    cls.urgency_floor_reasons = reasons
    cls.needs_human_triage = (cls.category is Category.unclear or cls.confidence < TRIAGE_CONFIDENCE
                              or any(a.confidence >= TRIAGE_ALT_CONFIDENCE for a in cls.alternatives))
    return cls


_KEYWORDS = [
    (Category.security_concern, r"breach|leak|hacked|unauthori[sz]ed|vulnerab|credential"),
    (Category.data_privacy, r"gdpr|personal data|delete my data|data deletion|privacy"),
    (Category.service_outage, r"\bdown\b|outage|unavailable|not working at all|cannot access|can't access"),
    (Category.billing_issue, r"invoice|charged|charge|refund|billing|payment"),
    (Category.churn_risk, r"cancel|switch(ing)? to|competitor|leaving|terminate our"),
    (Category.bug_report, r"bug|error|crash|fails?|broken|wrong"),
    (Category.feature_request, r"feature|would like|wish|please add|support for"),
    (Category.praise, r"thank|great|love|excellent|awesome"),
]
_ABUSIVE = re.compile(r"\b(idiot|stupid|useless|scam|garbage|incompetent|moron|damn)\b", re.I)


def keyword_classify(text: str) -> Classification:
    """Degraded path when the LLM is unavailable: coarse, never confident, always sent to triage."""
    category = next((c for c, rx in _KEYWORDS if re.search(rx, text, re.I)), Category.unclear)
    sentiment = (Sentiment.abusive if _ABUSIVE.search(text) else
                 Sentiment.positive if category is Category.praise else Sentiment.negative)
    cls = Classification(category=category, sentiment=sentiment, urgency=Urgency.medium, confidence=0.4,
                         rationale="keyword rules (LLM unavailable)", classified_by="keyword_rules")
    return apply_floor(cls, text)


def classify(caller: LLMCaller, fb: Feedback) -> Classification:
    """Raises LLMUnavailable when the provider fails or returns an unusable answer."""
    msgs = [Message("system", load_prompt(CLASSIFY_PROMPT)),
            Message("user", f"{wrap_feedback(fb.text)}\n\nChannel: {fb.channel.value}")]
    turn = caller.call("classify", CLASSIFY_PROMPT, msgs, [CLASSIFY_TOOL],
                       ToolChoice("any", ["classify_feedback"]))
    call = next((c for c in turn.tool_calls if c.name == "classify_feedback"), None)
    if call is None:
        raise LLMUnavailable("classify_feedback was not called")
    try:
        args = ClassifyArgs.model_validate(call.args)
    except ValidationError as e:
        raise LLMUnavailable(f"invalid classification: {e.error_count()} schema errors") from e
    cls = Classification(**args.model_dump(), classified_by="llm")
    return apply_floor(cls, fb.text)

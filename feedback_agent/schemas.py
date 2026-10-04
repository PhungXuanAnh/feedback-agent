"""Typed data contracts shared by every pipeline step."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

MAX_FEEDBACK_CHARS = 4000


class Category(str, Enum):
    billing_issue = "billing_issue"
    bug_report = "bug_report"
    service_outage = "service_outage"
    feature_request = "feature_request"
    praise = "praise"
    churn_risk = "churn_risk"
    abuse_policy_violation = "abuse_policy_violation"
    security_concern = "security_concern"
    data_privacy = "data_privacy"
    unclear = "unclear"


class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"
    abusive = "abusive"


class ActionType(str, Enum):
    issue_refund = "issue_refund"  # the only money action; fully checked by the validator
    escalate_to_engineering = "escalate_to_engineering"
    escalate_to_human = "escalate_to_human"
    request_more_info = "request_more_info"
    log_only = "log_only"


class Channel(str, Enum):
    email = "email"
    web_form = "web_form"
    chat = "chat"
    api = "api"


URGENCY_ORDER = [Urgency.low, Urgency.medium, Urgency.high, Urgency.critical]
TIER_ORDER = ["standard", "pro", "enterprise"]
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def max_urgency(a: Urgency, b: Urgency) -> Urgency:
    return a if URGENCY_ORDER.index(a) >= URGENCY_ORDER.index(b) else b


# --------------------------------------------------------------------- intake
class Feedback(BaseModel):
    """Free text plus minimal metadata (step 1: intake)."""

    text: str = Field(min_length=1)
    customer_email: str
    customer_id: Optional[str] = None
    channel: Channel = Channel.web_form
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feedback_id: Optional[str] = None
    truncated: bool = False

    @field_validator("customer_email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("customer_email is not a valid email address")
        return v

    @field_validator("text")
    @classmethod
    def _text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank")
        return v

    def model_post_init(self, __context: Any) -> None:
        if len(self.text) > MAX_FEEDBACK_CHARS:
            self.text = self.text[:MAX_FEEDBACK_CHARS]
            self.truncated = True

    @property
    def as_of(self):
        """Policy validity date: server-issued, equals the time feedback was received."""
        return self.received_at.date()


# ------------------------------------------------------------- classification
class Alternative(BaseModel):
    category: Category
    confidence: float = Field(ge=0, le=1)


class Classification(BaseModel):
    """What the LLM returns from classify_feedback (plus code-owned fields)."""

    category: Category
    alternatives: list[Alternative] = Field(default_factory=list, max_length=2)
    sentiment: Sentiment
    urgency: Urgency
    confidence: float = Field(ge=0, le=1)
    rationale: str = ""
    # code-owned
    classified_by: str = "llm"  # llm | keyword_rules
    needs_human_triage: bool = False
    urgency_uncertain: bool = False
    urgency_floor_reasons: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------- report
class CustomerContext(BaseModel):
    record_found: bool
    customer_id: Optional[str] = None
    tier: Optional[str] = None
    tenure_months: Optional[int] = None
    open_tickets: Optional[int] = None
    note: Optional[str] = None


class ActionParams(BaseModel):
    """Typed (not free-form) so the function schema is valid for Gemini and every field is checkable."""

    invoice_id: Optional[str] = Field(default=None, description="issue_refund: invoice from the customer record")
    amount: Optional[float] = Field(default=None, allow_inf_nan=False, description="issue_refund: amount in USD")
    target: Optional[str] = Field(default=None, description="escalations: team or queue, e.g. 'engineering_oncall'")


class SuggestedAction(BaseModel):
    type: ActionType
    params: ActionParams = Field(default_factory=ActionParams)
    basis: list[str] = Field(default_factory=list)
    rationale: str = ""


class ReportDraft(BaseModel):
    """Arguments of the submit_report tool: everything the LLM is allowed to write."""

    summary: str = Field(max_length=900)
    secondary_categories: list[Category] = Field(default_factory=list, max_length=3)
    customer_context: "DraftCustomerContext"
    cited_policies: list[str] = Field(default_factory=list)
    cited_guidelines: list[str] = Field(default_factory=list)
    suggested_actions: list[SuggestedAction] = Field(min_length=1)
    draft_clarification_question: Optional[str] = None


class DraftCustomerContext(BaseModel):
    record_found: bool
    customer_id: Optional[str] = None
    tier: Optional[str] = None


ReportDraft.model_rebuild()

ConfidenceLevel = Literal["low", "medium", "high"]


class Report(BaseModel):
    """The structured feedback report handed to the CS officer."""

    report_id: str
    feedback_id: str
    trace_id: str
    status: str = "pending_review"
    summary: str
    category: Category
    urgency: Urgency
    sentiment: Sentiment
    alternatives: list[Alternative] = Field(default_factory=list)
    secondary_categories: list[Category] = Field(default_factory=list)
    customer_context: CustomerContext
    cited_policies: list[str] = Field(default_factory=list)
    cited_guidelines: list[str] = Field(default_factory=list)
    suggested_actions: list[SuggestedAction]
    draft_clarification_question: Optional[str] = None
    flags: list[str] = Field(default_factory=list)
    needs_human_triage: bool = False
    confidence_score: float
    confidence_level: ConfidenceLevel
    prompt_versions: dict[str, str]
    report_generated_by: str  # provider name (gemini | scripted | ...) or degraded_template
    provider: str
    model: str
    as_of: str
    counters: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

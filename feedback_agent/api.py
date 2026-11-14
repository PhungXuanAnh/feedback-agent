"""FastAPI surface: intake + the review queue for CS officers (no UI by design)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, ValidationError

from .apiguard import ApiGuard
from .bootstrap import build_pipeline, open_database
from .config import Settings
from .database import StorageError
from .hitl import ReviewError, ReviewRequest, ReviewService, pending_summary
from .llm import ConfigError
from .pipeline import FeedbackPipeline
from .schemas import Category, Channel, Feedback, Report, SuggestedAction, Urgency
from .trace import load_trace_by_id


class FeedbackIn(BaseModel):
    text: str
    customer_email: str
    customer_id: Optional[str] = None
    channel: Channel = Channel.web_form
    received_at: Optional[datetime] = None  # default: now; also fixes which policies are in force


class ErrorMessage(BaseModel):
    detail: str


class ReportSummary(BaseModel):
    report_id: str
    feedback_id: str
    status: str
    category: Category
    urgency: Urgency
    confidence_level: str
    needs_human_triage: bool
    report_generated_by: str
    created_at: str


class ReportSnapshot(BaseModel):
    """The fields a CS officer can override."""

    category: Category
    urgency: Urgency
    suggested_actions: list[SuggestedAction]
    draft_clarification_question: Optional[str] = None


class ReviewOverrides(BaseModel):
    before: ReportSnapshot
    after: ReportSnapshot


class ReviewRecord(BaseModel):
    id: int
    report_id: str
    actor: str
    decision: Literal["approve", "override", "reject"]
    overrides: Optional[ReviewOverrides] = None
    note: Optional[str] = None
    at: str
    machine_category: Optional[str] = None
    machine_urgency: Optional[str] = None


class ReportDetail(BaseModel):
    report: Report
    status: str
    executed_at: Optional[str] = None
    reviews: list[ReviewRecord]
    machine_suggestion_kept: bool
    final: Optional[ReportSnapshot] = None


class ReviewResult(BaseModel):
    report_id: str
    decision: Literal["approve", "override", "reject"]
    status: str
    executed: bool
    idempotent: bool
    execution: Optional[list[str]] = None  # stub executor calls; null when rejected or replayed


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="allow")  # every event type adds its own fields

    seq: int
    ts: str
    trace_id: str
    event: str


UNAUTHORIZED = {
    401: {"model": ErrorMessage, "description": "Missing or invalid X-API-Token header"}
}
NOT_FOUND = {404: {"model": ErrorMessage, "description": "Not found"}}


def create_app(
    settings: Optional[Settings] = None, pipeline: Optional[FeedbackPipeline] = None
) -> FastAPI:
    """The FastAPI app. Tests pass `pipeline`; otherwise it is built on the first POST /feedback,
    so the app starts (and /health answers) even when the LLM config is missing.
    """
    settings = settings or Settings.from_env()
    app = FastAPI(title="Feedback agent", version="0.1.0")
    state: dict[str, Any] = {"pipeline": pipeline}
    database = pipeline.database if pipeline else open_database(settings)
    reviews = ReviewService(database, settings)
    guard = ApiGuard(settings, database)
    protected = [Depends(guard.auth)]  # /health and the API docs stay open

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness check; needs no token."""
        return {"status": "ok"}

    @app.post(
        "/feedback",
        status_code=201,
        dependencies=[Depends(guard.auth), Depends(guard.cost_guard)],
        responses={
            **UNAUTHORIZED,
            429: {"model": ErrorMessage, "description": "Rate limit or daily cap reached"},
            503: {
                "model": ErrorMessage,
                "description": "Report not queued: database or LLM config unavailable",
            },
        },
    )
    def submit_feedback(body: FeedbackIn) -> Report:
        """Run the feedback through the pipeline and return the report, saved as pending_review."""
        try:
            if state["pipeline"] is None:
                state["pipeline"] = build_pipeline(settings)
            try:
                feedback = Feedback(**body.model_dump(exclude_none=True))
            except ValidationError as e:  # same 422 shape as FastAPI's own body validation
                raise RequestValidationError(
                    [{**x, "loc": ("body", *x["loc"])} for x in e.errors(include_url=False)]
                ) from e
            return state["pipeline"].process(feedback)
        except (
            StorageError,
            ConfigError,
        ) as e:  # never claim the report was queued when it was not
            raise HTTPException(503, f"report not queued: {e}") from e

    @app.get(
        "/reports",
        dependencies=protected,
        response_model=list[ReportSummary],
        responses=UNAUTHORIZED,
    )
    def list_reports(status: Optional[str] = None) -> list[dict]:
        """List reports, optionally of one status (e.g. pending_review)."""
        return [pending_summary(row) for row in database.list_reports(status)]

    @app.get(
        "/reports/{report_id}",
        dependencies=protected,
        response_model=ReportDetail,
        responses={**UNAUTHORIZED, **NOT_FOUND},
    )
    def get_report(report_id: str) -> dict:
        """One report with its review status and audit trail."""
        try:
            return reviews.get(report_id)
        except ReviewError as e:
            raise HTTPException(e.status_code, str(e)) from e

    @app.post(
        "/reports/{report_id}/review",
        dependencies=protected,
        response_model=ReviewResult,
        responses={
            **UNAUTHORIZED,
            **NOT_FOUND,
            409: {"model": ErrorMessage, "description": "Report already decided differently"},
            422: {
                "description": "Invalid body, or the override failed the action validator",
                "content": {
                    "application/json": {
                        "schema": {
                            "anyOf": [
                                {"$ref": "#/components/schemas/HTTPValidationError"},
                                {"$ref": "#/components/schemas/ErrorMessage"},
                            ]
                        }
                    }
                },
            },
        },
    )
    def review(report_id: str, review_request: ReviewRequest) -> dict:
        """Record the officer's decision: approve, override (with changes) or reject."""
        try:
            return reviews.review(report_id, review_request)
        except ReviewError as e:
            raise HTTPException(e.status_code, str(e)) from e

    @app.get(
        "/traces/{trace_id}",
        dependencies=protected,
        response_model=list[TraceEvent],
        responses={**UNAUTHORIZED, **NOT_FOUND},
    )
    def trace(trace_id: str) -> list[dict]:
        """Every step of the run that produced a report, by trace id."""
        try:
            return load_trace_by_id(trace_id, settings.trace_dir)
        except FileNotFoundError:
            raise HTTPException(404, "trace not found") from None

    app.state.reviews = reviews
    return app

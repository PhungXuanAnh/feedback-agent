"""FastAPI surface: intake + the review queue for CS officers (no UI by design)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from .bootstrap import build_pipeline, open_store
from .config import Settings
from .hitl import ReviewError, ReviewRequest, ReviewService, pending_summary
from .llm import ConfigError
from .pipeline import FeedbackPipeline
from .schemas import Channel, Feedback
from .store import StorageError
from .trace import load_trace_by_id


class FeedbackIn(BaseModel):
    text: str
    customer_email: str
    customer_id: Optional[str] = None
    channel: Channel = Channel.web_form
    received_at: Optional[datetime] = None  # default: now; also fixes which policies are in force


def create_app(settings: Optional[Settings] = None, pipeline: Optional[FeedbackPipeline] = None) -> FastAPI:
    s = settings or Settings.from_env()
    app = FastAPI(title="Feedback agent", version="0.1.0")
    state: dict[str, Any] = {"pipeline": pipeline}
    store = pipeline.store if pipeline else open_store(s)
    reviews = ReviewService(store, s)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/feedback", status_code=201)
    def submit_feedback(body: FeedbackIn) -> dict:
        try:
            if state["pipeline"] is None:
                state["pipeline"] = build_pipeline(s)
            try:
                fb = Feedback(**body.model_dump(exclude_none=True))
            except ValidationError as e:
                raise HTTPException(422, [{"field": ".".join(map(str, x["loc"])), "error": x["msg"]}
                                          for x in e.errors()]) from e
            return state["pipeline"].process(fb).model_dump(mode="json")
        except (StorageError, ConfigError) as e:  # never claim the report was queued when it was not
            raise HTTPException(503, f"report not queued: {e}") from e

    @app.get("/reports")
    def list_reports(status: Optional[str] = None) -> list[dict]:
        return [pending_summary(r) for r in store.list_reports(status)]

    @app.get("/reports/{report_id}")
    def get_report(report_id: str) -> dict:
        try:
            return reviews.get(report_id)
        except ReviewError as e:
            raise HTTPException(e.status_code, str(e)) from e

    @app.post("/reports/{report_id}/review")
    def review(report_id: str, req: ReviewRequest) -> dict:
        try:
            return reviews.review(report_id, req)
        except ReviewError as e:
            raise HTTPException(e.status_code, str(e)) from e

    @app.get("/traces/{trace_id}")
    def trace(trace_id: str) -> list[dict]:
        try:
            return load_trace_by_id(trace_id, s.trace_dir)
        except FileNotFoundError:
            raise HTTPException(404, "trace not found") from None

    app.state.reviews = reviews
    return app

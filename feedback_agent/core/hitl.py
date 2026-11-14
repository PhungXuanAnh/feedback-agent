"""Step 6: human-in-the-loop. Reports wait in `pending_review`; only a human decision moves them,
and only approved/overridden reports reach the executor (a stub that prints the API call it would make).

States: pending_review -> approved | overridden -> executed ; pending_review -> rejected (terminal).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ..config import Settings
from ..database import Database
from ..ops.trace import Event, Tracer
from ..schemas import ActionType, Category, Report, SuggestedAction, Urgency
from .evidence import Evidence
from .validator import validate_actions

log = logging.getLogger("feedback_agent.executor")


class ReviewError(Exception):
    status_code = 400


class ReportNotFound(ReviewError):
    status_code = 404


class ReviewConflict(ReviewError):
    status_code = 409


class OverrideRejected(ReviewError):
    status_code = 422


class Overrides(BaseModel):
    category: Optional[Category] = None
    urgency: Optional[Urgency] = None
    suggested_actions: Optional[list[SuggestedAction]] = None
    draft_clarification_question: Optional[str] = None


class ReviewRequest(BaseModel):
    decision: Literal["approve", "override", "reject"]
    actor: str = Field(min_length=1, max_length=80)
    note: str = Field(default="", max_length=1000)
    overrides: Optional[Overrides] = None


class StubExecutor:
    """Stand-in for the real ticketing/billing APIs: records and logs the call it WOULD make."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, report_id: str, actions: list[SuggestedAction]) -> list[str]:
        """The API call each action would make, logged and returned.

        Example: issue_refund -> 'POST /billing/refunds {"invoice_id": ..., "amount": ..., ...}'.
        """
        executed_calls = []
        for action in actions:
            params = action.params.model_dump(exclude_none=True)
            body = json.dumps({**params, "report_id": report_id})
            call = {
                ActionType.issue_refund: f"POST /billing/refunds {body}",
                ActionType.escalate_to_engineering: f"POST /escalations/engineering {body}",
                ActionType.escalate_to_human: f"POST /escalations/human {body}",
                ActionType.request_more_info: f"TASK officer: send clarification question for {report_id} (system never contacts the customer)",
                ActionType.log_only: f"POST /feedback-log {json.dumps({'report_id': report_id})}",
            }[action.type]
            log.info("[stub executor] %s", call)
            executed_calls.append(call)
        self.calls += executed_calls
        return executed_calls


def _reviewable_fields(report: dict, overrides: Optional[Overrides] = None) -> dict:
    """The reviewable part of a report (category, urgency, actions, question), overrides applied.

    Saved as before/after in the audit row, so the machine's suggestion stays next to the final one.
    """
    snapshot = {
        "category": report["category"],
        "urgency": report["urgency"],
        "suggested_actions": report["suggested_actions"],
        "draft_clarification_question": report.get("draft_clarification_question"),
    }
    if overrides:
        for k, v in overrides.model_dump(mode="json", exclude_none=True).items():
            snapshot[k] = v
    return snapshot


class ReviewService:
    """Applies an officer's decision to a pending report and runs the approved actions."""

    def __init__(
        self, database: Database, settings: Settings, action_executor: Optional[StubExecutor] = None
    ):
        self.database, self.settings, self.action_executor = (
            database,
            settings,
            action_executor or StubExecutor(),
        )

    def get(self, report_id: str) -> dict:
        """Report, status and audit rows; `final` is the officer's version after an override."""
        row = self.database.get_report_row(report_id)
        if not row:
            raise ReportNotFound(report_id)
        report = json.loads(row["payload_json"])
        reviews = [
            {
                **review,
                "overrides": (
                    json.loads(review["overrides_json"]) if review["overrides_json"] else None
                ),
            }
            for review in self.database.reviews_for(report_id)
        ]
        for review in reviews:
            review.pop("overrides_json")
        final = (
            reviews[0]["overrides"]["after"]
            if reviews and reviews[0]["decision"] == "override"
            else None
        )
        return {
            "report": report,
            "status": row["status"],
            "executed_at": row["executed_at"],
            "reviews": reviews,
            "machine_suggestion_kept": True,
            "final": final,
        }

    def review(self, report_id: str, review_request: ReviewRequest) -> dict:
        """Apply approve / override / reject to a pending report and return the outcome.

        An override must pass validate_actions against the saved evidence (else 422). The first
        decision wins: the same one again is an idempotent replay, a different one -> 409.
        Approved or overridden actions then go to the executor and the report becomes executed.
        """
        row = self.database.get_report_row(report_id)
        if not row:
            raise ReportNotFound(report_id)
        report = json.loads(row["payload_json"])
        overrides_json, after = None, None
        if review_request.decision == "override":
            if not review_request.overrides or not review_request.overrides.model_dump(
                exclude_none=True
            ):
                raise OverrideRejected("override needs at least one change")
            after = _reviewable_fields(report, review_request.overrides)
            # the same validator that checked the machine's report checks the officer's final actions
            actions = [
                SuggestedAction.model_validate(action) for action in after["suggested_actions"]
            ]
            issues = validate_actions(
                actions,
                Evidence.from_dict(json.loads(row["evidence_json"])),
                after.get("draft_clarification_question"),
            )
            if issues:
                raise OverrideRejected("; ".join(issues))
            overrides_json = json.dumps({"before": _reviewable_fields(report), "after": after})
        new_status = {"approve": "approved", "override": "overridden", "reject": "rejected"}[
            review_request.decision
        ]
        applied = self.database.apply_review(
            report_id,
            new_status,
            {
                "actor": review_request.actor,
                "decision": review_request.decision,
                "overrides_json": overrides_json,
                "note": review_request.note,
                "at": datetime.now(timezone.utc).isoformat(),
                "machine_category": report["category"],
                "machine_urgency": report["urgency"],
            },
        )
        if (
            not applied
        ):  # somebody decided first: same decision = replay of the old result, otherwise 409
            first = self.database.reviews_for(report_id)[0]
            if (
                first["decision"] == review_request.decision
                and first["overrides_json"] == overrides_json
            ):
                return {
                    **self._decision_result(report_id, review_request.decision),
                    "idempotent": True,
                }
            raise ReviewConflict(f"report already {first['decision']} by {first['actor']}")
        execution = None
        if review_request.decision != "reject":
            final = after["suggested_actions"] if after else report["suggested_actions"]
            final_actions = [SuggestedAction.model_validate(action) for action in final]
            execution = self.action_executor.execute(report_id, final_actions)
            self.database.mark_executed(report_id, datetime.now(timezone.utc).isoformat())
        status = self.database.get_report_row(report_id)["status"]
        Tracer(report["trace_id"], self.settings).emit(
            Event.human_review,
            report_id=report_id,
            decision=review_request.decision,
            actor=review_request.actor,
            status=status,
        )
        return {
            **self._decision_result(report_id, review_request.decision),
            "idempotent": False,
            "execution": execution,
        }

    def _decision_result(self, report_id: str, decision: str) -> dict[str, Any]:
        """Short outcome of a decision: report id, decision, current status, executed or not."""
        row = self.database.get_report_row(report_id)
        return {
            "report_id": report_id,
            "decision": decision,
            "status": row["status"],
            "executed": row["executed_at"] is not None,
        }


def report_summary(row: dict) -> dict:
    """One row of the review queue: the fields an officer needs to pick a report."""
    report = json.loads(row["payload_json"])
    return {
        "report_id": row["id"],
        "feedback_id": row["feedback_id"],
        "status": row["status"],
        "category": report["category"],
        "urgency": report["urgency"],
        "confidence_level": report["confidence_level"],
        "needs_human_triage": report["needs_human_triage"],
        "report_generated_by": report["report_generated_by"],
        "created_at": row["created_at"],
    }


__all__ = ["ReviewService", "ReviewRequest", "Overrides", "StubExecutor", "ReviewError", "Report"]

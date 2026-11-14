"""The fixed six-step pipeline: intake -> guard -> classify -> agent loop -> report+validator -> HITL queue.
Only classify, the agent loop and submit_report use the LLM; everything else is plain code."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from ..config import Settings
from ..database import Database
from ..llm.base import LLMProvider
from ..llm.caller import LLMCaller, LLMUnavailable
from ..ops.trace import Event, Tracer, new_trace_id
from ..schemas import Classification, Feedback, Report, ReportDraft
from . import guard
from .agent import AgentLoop, policy_query
from .classify import classify, classify_by_keywords
from .evidence import SYSTEM_GUIDELINE, Evidence
from .prompts import AGENT_PROMPT, CLASSIFY_PROMPT
from .report import build_report
from .tool_executor import RunState, ToolExecutor
from .tools import FaultInjector, KnowledgeBase, ToolContext

PROMPT_VERSIONS = {"classify": CLASSIFY_PROMPT, "agent": AGENT_PROMPT}
_TEMPLATE_REASONS = {
    "llm_unavailable": "LLM unavailable after retry",
    "grounding_failed": "grounding check failed twice",
    "budget_exhausted": "step cap reached without a valid report",
    "incomplete_context": "a required source was unavailable",
}
_OUTCOME_FLAGS = {
    "grounding_failed": "grounding_failed",
    "budget_exhausted": "step_limit_reached",
    "llm_unavailable": "llm_unavailable",
}


@dataclass
class DraftOutcome:
    """What steps 4-5 produced: the LLM's draft, or None and why the template report is used."""

    draft: Optional[ReportDraft]
    generated_by: str
    template_reason: str
    failed_sources: dict[str, str]
    llm_turns: int


class FeedbackPipeline:
    """All steps for one feedback; holds what requests share (database, KB, provider)."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        knowledge_base: KnowledgeBase,
        provider: LLMProvider,
        fault_injector: Optional[FaultInjector] = None,
    ):
        self.settings, self.database, self.knowledge_base, self.provider = (
            settings,
            database,
            knowledge_base,
            provider,
        )
        self.fault_injector = fault_injector or FaultInjector()

    def process(self, feedback: Feedback) -> Report:
        """Run one feedback through the six steps and return the saved report.

        The API or CLI has already validated the input (HTTP 422 otherwise).
        1. Intake: give the feedback an id and open its trace.
        2. Injection guard: regexes flag suspicious text; nothing is blocked.
        3. Classify: one schema-forced LLM call plus urgency floor rules; keyword rules when the
           LLM is unavailable.
        4. Agent loop: the LLM calls the three retrieval tools and submits a draft report.
        5. Report + validator: the draft must pass the grounding validator (one repair allowed);
           without a valid draft, the degraded path fetches the missing sources by code and a
           template is used. Code then adds classification, flags and computed confidence.
        6. Review queue: the report is saved as pending_review; an officer decides later
           through the API or CLI (hitl.py).

        StorageError propagates, so a caller never reports "queued" when saving failed.
        """
        # 1. Intake
        tracer = self._start_trace(feedback)
        # 2. Injection guard
        flags = self._scan_for_injection(feedback, tracer)
        # 3. Classify
        llm_caller = LLMCaller(self.provider, self.settings, tracer)
        classification, llm_ok = self._classify(llm_caller, feedback, tracer)
        # 4. Agent loop, and the validator half of step 5 (or the degraded path)
        run_state = self._new_run_state(feedback, classification, flags, tracer)
        tool_executor = ToolExecutor(
            run_state, self.settings.max_tool_calls, self.settings.max_forced_lookups
        )
        outcome = self._draft_report(run_state, tool_executor, llm_caller, llm_ok)
        # 5. Report
        report = self._build_report(run_state, tool_executor, outcome)
        # 6. Review queue
        self._save(run_state, report)
        return report

    def _start_trace(self, feedback: Feedback) -> Tracer:
        """Step 1, intake: give the feedback an id and open its trace."""
        tracer = Tracer(new_trace_id(), self.settings)
        feedback.feedback_id = feedback.feedback_id or "fb_" + uuid.uuid4().hex[:10]
        tracer.emit(
            Event.intake,
            feedback_id=feedback.feedback_id,
            channel=feedback.channel.value,
            chars=len(feedback.text),
            truncated=feedback.truncated,
            received_at=feedback.received_at.isoformat(),
            provider=self.provider.name,
            model=self.provider.model,
            max_llm_turns=self.settings.max_llm_turns,
            max_tool_calls=self.settings.max_tool_calls,
            prompt_versions=PROMPT_VERSIONS,
        )
        return tracer

    @staticmethod
    def _scan_for_injection(feedback: Feedback, tracer: Tracer) -> set[str]:
        """Step 2, injection guard: flag (never block) known injection patterns."""
        flags: set[str] = set()
        injection_patterns = guard.find_injection_patterns(feedback.text)
        if injection_patterns:
            flags.add("injection_suspected")
            tracer.emit(Event.injection_guard, source="feedback", patterns=injection_patterns)
        return flags

    @staticmethod
    def _classify(
        llm_caller: LLMCaller, feedback: Feedback, tracer: Tracer
    ) -> tuple[Classification, bool]:
        """Step 3, classify; falls back to keyword rules. Returns (classification, llm_ok)."""
        try:
            classification = classify(llm_caller, feedback)
            llm_ok = True
        except LLMUnavailable as e:
            classification = classify_by_keywords(feedback.text)
            llm_ok = False
            tracer.emit(
                Event.degraded_path, step="classify", reason=str(e)[:200], fallback="keyword_rules"
            )
        tracer.emit(Event.classification_completed, **classification.model_dump(mode="json"))
        return classification, llm_ok

    def _new_run_state(
        self, feedback: Feedback, classification: Classification, flags: set[str], tracer: Tracer
    ) -> RunState:
        """What the agent loop and the degraded path share: tool context, evidence, flags."""
        tool_context = ToolContext(
            self.database,
            self.knowledge_base,
            feedback.customer_email,
            feedback.customer_id,
            feedback.as_of,
            self.fault_injector,
        )
        evidence = Evidence(as_of=feedback.as_of)
        triage_guideline = self.knowledge_base.guideline_by_id(SYSTEM_GUIDELINE)
        if (
            triage_guideline
        ):  # system evidence: citable for escalate/request_more_info only; not one of the 3 sources
            evidence.system_evidence[SYSTEM_GUIDELINE] = triage_guideline
            tracer.emit(
                Event.system_evidence,
                guideline_id=SYSTEM_GUIDELINE,
                allowed_actions=triage_guideline["allowed_actions"],
            )
        return RunState(feedback, classification, tool_context, evidence, tracer, flags)

    def _draft_report(
        self,
        run_state: RunState,
        tool_executor: ToolExecutor,
        llm_caller: LLMCaller,
        llm_ok: bool,
    ) -> DraftOutcome:
        """Steps 4-5: the agent loop drafts a validated report; if not, the degraded path runs."""
        if llm_ok:
            agent_result = AgentLoop(run_state, tool_executor, llm_caller, self.settings).run()
            draft, llm_turns = agent_result.draft, agent_result.llm_turns
            failed_sources = agent_result.failed_sources
            template_reason = _TEMPLATE_REASONS.get(agent_result.outcome, "")
            if agent_result.outcome in _OUTCOME_FLAGS:
                run_state.flags.add(_OUTCOME_FLAGS[agent_result.outcome])
        else:
            draft, llm_turns, failed_sources = None, 0, {}
            template_reason = _TEMPLATE_REASONS["llm_unavailable"]
            run_state.flags.add("llm_unavailable")
        if draft is not None:
            return DraftOutcome(
                draft, self.provider.name, template_reason, failed_sources, llm_turns
            )
        # incomplete_context already knows which source is down: no point asking again
        if not failed_sources:
            failed_sources = self._run_degraded_lookups(
                run_state, tool_executor, run_state.tracer, template_reason
            )
        return DraftOutcome(None, "degraded_template", template_reason, failed_sources, llm_turns)

    def _build_report(
        self, run_state: RunState, tool_executor: ToolExecutor, outcome: DraftOutcome
    ) -> Report:
        """Step 5, report: code fills classification, flags and confidence around the draft."""
        tracer = run_state.tracer
        counters = {"llm_turns": outcome.llm_turns, **tool_executor.counters(), **tracer.totals()}
        report = build_report(
            feedback=run_state.feedback,
            classification=run_state.classification,
            evidence=run_state.evidence,
            flags=run_state.flags,
            draft=outcome.draft,
            generated_by=outcome.generated_by,
            provider=self.provider.name,
            model=self.provider.model,
            trace_id=tracer.trace_id,
            prompt_versions=PROMPT_VERSIONS,
            counters=counters,
            template_reason=outcome.template_reason,
            failed_sources=outcome.failed_sources,
        )
        tracer.emit(
            Event.report_generated,
            report_id=report.report_id,
            generated_by=outcome.generated_by,
            confidence=report.confidence_level,
            score=report.confidence_score,
            flags=report.flags,
            actions=[action.type.value for action in report.suggested_actions],
            counters=counters,
        )
        return report

    def _save(self, run_state: RunState, report: Report) -> None:
        """Step 6, review queue: save the feedback and the report as pending_review."""
        self.database.save_feedback(
            run_state.feedback,
            "injection_suspected" in run_state.flags,
            run_state.classification,
            CLASSIFY_PROMPT,
            run_state.tracer.trace_id,
        )
        self.database.save_report(
            report, run_state.evidence.to_dict()
        )  # StorageError propagates: never claim "queued"

    @staticmethod
    def _run_degraded_lookups(
        run_state: RunState, tool_executor: ToolExecutor, tracer: Tracer, reason: str
    ) -> dict:
        """No-LLM path: keep everything that already succeeded, call only the missing sources.

        Returns {source: error} for the sources still missing afterwards.
        """
        evidence = run_state.evidence
        plan = {
            "guidelines": (
                "get_cs_guidelines",
                {"category": run_state.classification.category.value},
            ),
            "customer": ("lookup_customer", {"email": run_state.feedback.customer_email}),
            "policies": (
                "search_policies",
                {
                    "query": policy_query(run_state.classification, run_state.feedback),
                    "category": run_state.classification.category.value,
                },
            ),
        }
        missing = list(evidence.missing_sources)
        tracer.emit(
            Event.degraded_path,
            step="agent",
            reason=reason,
            reused=sorted(evidence.consulted),
            missing=missing,
        )
        for source in missing:
            name, args = plan[source]
            tool_executor.call(name, args, origin="degraded")
        return {
            source: tool_executor.failed_sources.get(source)
            or evidence.source_errors.get(source, "error")
            for source in evidence.missing_sources
        }

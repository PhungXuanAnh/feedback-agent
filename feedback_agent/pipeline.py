"""The fixed six-step pipeline: intake -> guard -> classify -> agent loop -> report+validator -> HITL queue.
Only classify, the agent loop and submit_report use the LLM; everything else is plain code."""
from __future__ import annotations

import uuid
from typing import Optional

from . import guard
from .agent import AgentLoop, RunState, ToolExecutor, policy_query
from .classify import classify, keyword_classify
from .config import Settings
from .evidence import SYSTEM_GUIDELINE, Evidence
from .llm.base import LLMProvider
from .llmcaller import LLMCaller, LLMUnavailable
from .prompts import AGENT_PROMPT, CLASSIFY_PROMPT
from .report import build_report
from .schemas import Feedback, Report
from .store import Store
from .tools import FaultInjector, KnowledgeBase, ToolContext
from .trace import Event, Tracer, new_trace_id

PROMPT_VERSIONS = {"classify": CLASSIFY_PROMPT, "agent": AGENT_PROMPT}


class FeedbackPipeline:
    def __init__(self, settings: Settings, store: Store, knowledge_base: KnowledgeBase, provider: LLMProvider,
                 fault_injector: Optional[FaultInjector] = None):
        self.settings, self.store, self.knowledge_base, self.provider = settings, store, knowledge_base, provider
        self.fault_injector = fault_injector or FaultInjector()

    def process(self, feedback: Feedback) -> Report:
        settings, tracer = self.settings, Tracer(new_trace_id(), self.settings)
        feedback.feedback_id = feedback.feedback_id or "fb_" + uuid.uuid4().hex[:10]
        tracer.emit(Event.intake, feedback_id=feedback.feedback_id, channel=feedback.channel.value,
                    chars=len(feedback.text),
                    truncated=feedback.truncated, received_at=feedback.received_at.isoformat(),
                    provider=self.provider.name,
                    model=self.provider.model, max_llm_turns=settings.max_llm_turns,
                    max_tool_calls=settings.max_tool_calls,
                    prompt_versions=PROMPT_VERSIONS)
        flags: set[str] = set()
        injection_patterns = guard.scan(feedback.text)
        if injection_patterns:
            flags.add("injection_suspected")
            tracer.emit(Event.injection_guard, source="feedback", patterns=injection_patterns)

        llm_caller = LLMCaller(self.provider, settings, tracer)
        llm_ok = True
        try:
            classification = classify(llm_caller, feedback)
        except LLMUnavailable as e:
            llm_ok = False
            classification = keyword_classify(feedback.text)
            tracer.emit(Event.degraded_path, step="classify", reason=str(e)[:200], fallback="keyword_rules")
        tracer.emit(Event.classification_completed, **classification.model_dump(mode="json"))

        tool_context = ToolContext(self.store, self.knowledge_base, feedback.customer_email, feedback.customer_id,
                                   feedback.as_of, self.fault_injector)
        evidence = Evidence(as_of=feedback.as_of)
        triage_guideline = self.knowledge_base.guideline_by_id(SYSTEM_GUIDELINE)
        if triage_guideline:  # system evidence: citable for escalate/request_more_info only; not one of the 3 sources
            evidence.system_evidence[SYSTEM_GUIDELINE] = triage_guideline
            tracer.emit(Event.system_evidence, guideline_id=SYSTEM_GUIDELINE,
                        allowed_actions=triage_guideline["allowed_actions"])
        run_state = RunState(feedback, classification, tool_context, evidence, tracer, flags)
        tool_executor = ToolExecutor(run_state, settings.max_tool_calls, settings.max_forced_lookups)

        draft, failed, rejected, reason, turns = None, {}, None, "", 0
        generated_by = self.provider.name
        if llm_ok:
            agent_result = AgentLoop(run_state, tool_executor, llm_caller, settings).loop()
            draft, turns, failed = agent_result.draft, agent_result.llm_turns, agent_result.failed_sources
            reason = {"llm_unavailable": "LLM unavailable after retry", "grounding_failed": "grounding check failed twice",
                      "budget_exhausted": "step cap reached without a valid report",
                      "incomplete_context": "a required source was unavailable"}.get(agent_result.outcome, "")
            if agent_result.outcome == "grounding_failed":
                flags.add("grounding_failed")
            if agent_result.outcome == "budget_exhausted":
                flags.add("step_limit_reached")
            if agent_result.outcome == "llm_unavailable" and "llm_unavailable" not in flags:
                flags.add("llm_unavailable")
        else:
            reason = "LLM unavailable after retry"
            flags.add("llm_unavailable")
        if draft is None:
            if not failed:  # incomplete_context already knows which source is down: no point asking again
                failed = self._degraded_lookups(run_state, tool_executor, tracer, reason)
            generated_by = "degraded_template"

        counters = {"llm_turns": turns, **tool_executor.counters(), **tracer.totals()}
        report = build_report(feedback=feedback, classification=classification, evidence=evidence, flags=flags,
                              draft=draft, generated_by=generated_by,
                              provider=self.provider.name, model=self.provider.model, trace_id=tracer.trace_id,
                              prompt_versions=PROMPT_VERSIONS, counters=counters, template_reason=reason,
                              failed_sources=failed)
        tracer.emit(Event.report_generated, report_id=report.report_id, generated_by=generated_by,
                    confidence=report.confidence_level, score=report.confidence_score, flags=report.flags,
                    actions=[action.type.value for action in report.suggested_actions], counters=counters)
        self.store.save_feedback(feedback, "injection_suspected" in flags, classification, CLASSIFY_PROMPT,
                                 tracer.trace_id)
        self.store.save_report(report, evidence.to_dict())  # StorageError propagates: never claim "queued"
        return report

    @staticmethod
    def _degraded_lookups(run_state: RunState, tool_executor: ToolExecutor, tracer: Tracer, reason: str) -> dict:
        """No-LLM path: keep everything that already succeeded, call only the missing sources."""
        evidence = run_state.evidence
        plan = {"guidelines": ("get_cs_guidelines", {"category": run_state.classification.category.value}),
                "customer": ("lookup_customer", {"email": run_state.feedback.customer_email}),
                "policies": ("search_policies", {"query": policy_query(run_state.classification, run_state.feedback),
                                                 "category": run_state.classification.category.value})}
        missing = list(evidence.missing_sources)
        tracer.emit(Event.degraded_path, step="agent", reason=reason, reused=sorted(evidence.consulted),
                    missing=missing)
        for source in missing:
            name, args = plan[source]
            tool_executor.call(name, args, origin="degraded")
        return {source: tool_executor.failed_sources.get(source) or evidence.source_errors.get(source, "error")
                for source in evidence.missing_sources}

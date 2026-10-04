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
    def __init__(self, settings: Settings, store: Store, kb: KnowledgeBase, provider: LLMProvider,
                 faults: Optional[FaultInjector] = None):
        self.s, self.store, self.kb, self.provider = settings, store, kb, provider
        self.faults = faults or FaultInjector()

    def process(self, fb: Feedback) -> Report:
        s, tr = self.s, Tracer(new_trace_id(), self.s)
        fb.feedback_id = fb.feedback_id or "fb_" + uuid.uuid4().hex[:10]
        tr.emit(Event.intake, feedback_id=fb.feedback_id, channel=fb.channel.value, chars=len(fb.text),
                truncated=fb.truncated, received_at=fb.received_at.isoformat(), provider=self.provider.name,
                model=self.provider.model, max_llm_turns=s.max_llm_turns, max_tool_calls=s.max_tool_calls,
                prompt_versions=PROMPT_VERSIONS)
        flags: set[str] = set()
        hits = guard.scan(fb.text)
        if hits:
            flags.add("injection_suspected")
            tr.emit(Event.injection_guard, source="feedback", patterns=hits)

        caller = LLMCaller(self.provider, s, tr)
        llm_ok = True
        try:
            cls = classify(caller, fb)
        except LLMUnavailable as e:
            llm_ok = False
            cls = keyword_classify(fb.text)
            tr.emit(Event.degraded_path, step="classify", reason=str(e)[:200], fallback="keyword_rules")
        tr.emit(Event.classification_completed, **cls.model_dump(mode="json"))

        ctx = ToolContext(self.store, self.kb, fb.customer_email, fb.customer_id, fb.as_of, self.faults)
        ev = Evidence(as_of=fb.as_of)
        triage = self.kb.guideline_by_id(SYSTEM_GUIDELINE)
        if triage:  # system evidence: citable for escalate/request_more_info only; not one of the 3 sources
            ev.system_evidence[SYSTEM_GUIDELINE] = triage
            tr.emit(Event.system_evidence, guideline_id=SYSTEM_GUIDELINE, allowed_actions=triage["allowed_actions"])
        st = RunState(fb, cls, ctx, ev, tr, flags)
        ex = ToolExecutor(st, s.max_tool_calls, s.max_forced_lookups)

        draft, failed, rejected, reason, turns = None, {}, None, "", 0
        generated_by = self.provider.name
        if llm_ok:
            res = AgentLoop(st, ex, caller, s).run()
            draft, turns, failed = res.draft, res.llm_turns, res.failed_sources
            reason = {"llm_unavailable": "LLM unavailable after retry", "grounding_failed": "grounding check failed twice",
                      "budget_exhausted": "step cap reached without a valid report",
                      "incomplete_context": "a required source was unavailable"}.get(res.outcome, "")
            if res.outcome == "grounding_failed":
                flags.add("grounding_failed")
            if res.outcome == "budget_exhausted":
                flags.add("step_limit_reached")
            if res.outcome == "llm_unavailable" and "llm_unavailable" not in flags:
                flags.add("llm_unavailable")
        else:
            reason = "LLM unavailable after retry"
            flags.add("llm_unavailable")
        if draft is None:
            if not failed:  # incomplete_context already knows which source is down: no point asking again
                failed = self._degraded_lookups(st, ex, tr, reason)
            generated_by = "degraded_template"

        counters = {"llm_turns": turns, **ex.counters(), **tr.totals()}
        report = build_report(fb=fb, cls=cls, ev=ev, flags=flags, draft=draft, generated_by=generated_by,
                              provider=self.provider.name, model=self.provider.model, trace_id=tr.trace_id,
                              prompt_versions=PROMPT_VERSIONS, counters=counters, template_reason=reason,
                              failed_sources=failed)
        tr.emit(Event.report_generated, report_id=report.report_id, generated_by=generated_by,
                confidence=report.confidence_level, score=report.confidence_score, flags=report.flags,
                actions=[a.type.value for a in report.suggested_actions], counters=counters)
        self.store.save_feedback(fb, "injection_suspected" in flags, cls, CLASSIFY_PROMPT, tr.trace_id)
        self.store.save_report(report, ev.to_dict())  # StorageError propagates: never claim "queued"
        return report

    @staticmethod
    def _degraded_lookups(st: RunState, ex: ToolExecutor, tr: Tracer, reason: str) -> dict:
        """No-LLM path: keep everything that already succeeded, call only the missing sources."""
        ev = st.ev
        plan = {"guidelines": ("get_cs_guidelines", {"category": st.cls.category.value}),
                "customer": ("lookup_customer", {"email": st.fb.customer_email}),
                "policies": ("search_policies", {"query": policy_query(st.cls, st.fb),
                                                 "category": st.cls.category.value})}
        missing = list(ev.missing_sources)
        tr.emit(Event.degraded_path, step="agent", reason=reason, reused=sorted(ev.consulted), missing=missing)
        for src in missing:
            name, args = plan[src]
            ex.call(name, args, origin="degraded")
        return {src: ex.failed.get(src) or ev.source_errors.get(src, "error") for src in ev.missing_sources}

"""Sample runs: each input in samples/inputs is processed in `live` (real Gemini) and/or `scripted`
(offline, no key) mode and stored with its report, trace and metadata under samples/<name>/."""
from __future__ import annotations

import json
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .bootstrap import build_pipeline
from .config import ROOT, Settings
from .llm import FaultInjectingProvider
from .pipeline import PROMPT_VERSIONS
from .schemas import URGENCY_ORDER, Feedback, Report, Urgency
from .tools import FaultInjector
from .trace import load_trace

SAMPLES = ROOT / "samples"


def load_manifest(root: Path = SAMPLES) -> list[dict]:
    return json.loads((root / "manifest.json").read_text("utf-8"))["samples"]


def check_expectations(report: Report, expect: dict[str, Any]) -> list[str]:
    """Return the list of unmet expectations (empty = the sample behaves as designed)."""
    action_types = {action.type.value for action in report.suggested_actions}
    unmet = []
    if "category" in expect and report.category.value != expect["category"]:
        unmet.append(f"category {report.category.value} != {expect['category']}")
    if ("min_urgency" in expect
            and URGENCY_ORDER.index(report.urgency) < URGENCY_ORDER.index(Urgency(expect["min_urgency"]))):
        unmet.append(f"urgency {report.urgency.value} < {expect['min_urgency']}")
    if "needs_human_triage" in expect and report.needs_human_triage != expect["needs_human_triage"]:
        unmet.append("needs_human_triage mismatch")
    if "customer_found" in expect and report.customer_context.record_found != expect["customer_found"]:
        unmet.append("customer_found mismatch")
    if "generated_by" in expect and report.report_generated_by != expect["generated_by"]:
        unmet.append(f"generated_by {report.report_generated_by}")
    if "generated_by_not" in expect and report.report_generated_by == expect["generated_by_not"]:
        unmet.append(f"generated_by is {report.report_generated_by}")
    unmet += [f"missing action {action}" for action in expect.get("actions_include", []) if action not in action_types]
    unmet += [f"forbidden action {action}" for action in expect.get("actions_forbid", []) if action in action_types]
    unmet += [f"missing flag {flag}" for flag in expect.get("flags_include", []) if flag not in report.flags]
    if expect.get("has_clarification_question") and not report.draft_clarification_question:
        unmet.append("no clarification question")
    return unmet


def run_one(sample: dict, mode: str, settings: Settings, root: Path = SAMPLES,
            out_dir: Optional[Path] = None) -> tuple[Report, dict]:
    """Run a single sample against a fresh temp DB/trace dir; write outputs when out_dir is given."""
    name, faults = sample["name"], sample.get("faults") or {}
    run_settings = deepcopy(settings)
    run_settings.llm_provider = "scripted" if mode == "scripted" else "gemini"
    run_settings.llm_retry_backoff_s = 0.2 if mode == "scripted" else settings.llm_retry_backoff_s
    with tempfile.TemporaryDirectory() as temp_dir:
        run_settings.database_path, run_settings.trace_dir = f"{temp_dir}/s.db", f"{temp_dir}/traces"
        script = root / "scripts" / f"{name}.json" if mode == "scripted" else None
        pipeline = build_pipeline(run_settings, script=str(script) if script else None,
                                  faults=_faults(faults.get("inject_tool_fault")))
        if mode == "live" and faults.get("inject_llm_fault"):  # scripted scripts contain their own failures
            pipeline.provider = FaultInjectingProvider(pipeline.provider, faults["inject_llm_fault"])
        feedback = Feedback(**json.loads((root / "inputs" / f"{name}.json").read_text("utf-8")))
        report = pipeline.process(feedback)
        problems = check_expectations(report, sample["expect"])
        meta = {"sample": name, "title": sample["title"], "mode": mode, "provider": pipeline.provider.name,
                "model": pipeline.provider.model, "prompt_versions": PROMPT_VERSIONS, "trace_id": report.trace_id,
                "report_generated_by": report.report_generated_by,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "counters": report.counters, "fault_injection": faults or None, "expectations_met": not problems,
                "unmet_expectations": problems, "notes": _notes(sample, mode, report)}
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(root / "inputs" / f"{name}.json", out_dir / "input.json")
            (out_dir / f"report.{mode}.json").write_text(report.model_dump_json(indent=2, exclude_none=True) + "\n", "utf-8")
            events = load_trace(report.trace_id, run_settings.trace_dir)
            (out_dir / f"trace.{mode}.jsonl").write_text(
                "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), "utf-8")
            (out_dir / f"metadata.{mode}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return report, meta


def _faults(specs: Optional[list[str]]) -> FaultInjector:
    injector = FaultInjector()
    for spec in specs or []:
        name, _, count = spec.partition(":")
        injector.tool_failures[name] = int(count) if count else 1
    return injector


def _notes(sample: dict, mode: str, report: Report) -> str:
    faults = sample.get("faults")
    if mode == "live" and faults:
        return ("NOT written by Gemini. Gemini was tried: it classified the message and made the first agent turn "
                "(tool calls succeeded). Failures were then injected on purpose (not a real outage): every LLM call "
                f"from call #{faults['inject_llm_fault']} on raised a transient error (retried once, failed again), "
                "and search_policies failed once (retried automatically, then succeeded). The report was built by the "
                "degraded template from the results that had already succeeded.")
    if mode == "scripted":
        return ("Replayed offline by ScriptedProvider with a hand-written script (samples/scripts). It shows the "
                "pipeline's behaviour, not LLM judgement; the live variant shows what Gemini decided.")
    return "Generated by the real LLM provider; the model chose which tools to call."


def run_samples(settings: Settings, modes: list[str], only: Optional[str] = None,
                root: Path = SAMPLES) -> list[dict]:
    metas = []
    for sample in load_manifest(root):
        if only and sample["name"] != only:
            continue
        for mode in modes:
            _, meta = run_one(sample, mode, settings, root, root / sample["name"])
            metas.append(meta)
    return metas

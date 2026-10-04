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


def check_expectations(rep: Report, expect: dict[str, Any]) -> list[str]:
    """Return the list of unmet expectations (empty = the sample behaves as designed)."""
    acts = {a.type.value for a in rep.suggested_actions}
    bad = []
    if "category" in expect and rep.category.value != expect["category"]:
        bad.append(f"category {rep.category.value} != {expect['category']}")
    if "min_urgency" in expect and URGENCY_ORDER.index(rep.urgency) < URGENCY_ORDER.index(Urgency(expect["min_urgency"])):
        bad.append(f"urgency {rep.urgency.value} < {expect['min_urgency']}")
    if "needs_human_triage" in expect and rep.needs_human_triage != expect["needs_human_triage"]:
        bad.append("needs_human_triage mismatch")
    if "customer_found" in expect and rep.customer_context.record_found != expect["customer_found"]:
        bad.append("customer_found mismatch")
    if "generated_by" in expect and rep.report_generated_by != expect["generated_by"]:
        bad.append(f"generated_by {rep.report_generated_by}")
    if "generated_by_not" in expect and rep.report_generated_by == expect["generated_by_not"]:
        bad.append(f"generated_by is {rep.report_generated_by}")
    bad += [f"missing action {a}" for a in expect.get("actions_include", []) if a not in acts]
    bad += [f"forbidden action {a}" for a in expect.get("actions_forbid", []) if a in acts]
    bad += [f"missing flag {f}" for f in expect.get("flags_include", []) if f not in rep.flags]
    if expect.get("has_clarification_question") and not rep.draft_clarification_question:
        bad.append("no clarification question")
    return bad


def run_one(sample: dict, mode: str, settings: Settings, root: Path = SAMPLES,
            out_dir: Optional[Path] = None) -> tuple[Report, dict]:
    """Run a single sample against a fresh temp DB/trace dir; write outputs when out_dir is given."""
    name, faults = sample["name"], sample.get("faults") or {}
    s = deepcopy(settings)
    s.llm_provider = "scripted" if mode == "scripted" else "gemini"
    s.llm_retry_backoff_s = 0.2 if mode == "scripted" else settings.llm_retry_backoff_s
    with tempfile.TemporaryDirectory() as tmp:
        s.database_path, s.trace_dir = f"{tmp}/s.db", f"{tmp}/traces"
        script = root / "scripts" / f"{name}.json" if mode == "scripted" else None
        pipe = build_pipeline(s, script=str(script) if script else None,
                              faults=_faults(faults.get("inject_tool_fault")))
        if mode == "live" and faults.get("inject_llm_fault"):  # scripted scripts contain their own failures
            pipe.provider = FaultInjectingProvider(pipe.provider, faults["inject_llm_fault"])
        fb = Feedback(**json.loads((root / "inputs" / f"{name}.json").read_text("utf-8")))
        report = pipe.process(fb)
        problems = check_expectations(report, sample["expect"])
        meta = {"sample": name, "title": sample["title"], "mode": mode, "provider": pipe.provider.name,
                "model": pipe.provider.model, "prompt_versions": PROMPT_VERSIONS, "trace_id": report.trace_id,
                "report_generated_by": report.report_generated_by, "generated_at": datetime.now(timezone.utc).isoformat(),
                "counters": report.counters, "fault_injection": faults or None, "expectations_met": not problems,
                "unmet_expectations": problems, "notes": _notes(sample, mode, report)}
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(root / "inputs" / f"{name}.json", out_dir / "input.json")
            (out_dir / f"report.{mode}.json").write_text(report.model_dump_json(indent=2, exclude_none=True) + "\n", "utf-8")
            (out_dir / f"trace.{mode}.jsonl").write_text(
                "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in load_trace(report.trace_id, s.trace_dir)), "utf-8")
            (out_dir / f"metadata.{mode}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return report, meta


def _faults(specs: Optional[list[str]]) -> FaultInjector:
    f = FaultInjector()
    for spec in specs or []:
        n, _, k = spec.partition(":")
        f.tool_failures[n] = int(k) if k else 1
    return f


def _notes(sample: dict, mode: str, rep: Report) -> str:
    f = sample.get("faults")
    if mode == "live" and f:
        return ("NOT written by Gemini. Gemini was tried: it classified the message and made the first agent turn "
                "(tool calls succeeded). Failures were then injected on purpose (not a real outage): every LLM call "
                f"from call #{f['inject_llm_fault']} on raised a transient error (retried once, failed again), and "
                "search_policies failed once (retried automatically, then succeeded). The report was built by the "
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

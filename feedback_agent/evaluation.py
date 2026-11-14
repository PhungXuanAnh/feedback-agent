"""Small eval harness: labelled cases run through the real pipeline, scored by code.

It reports measured numbers only (no targets): with ~14 cases the figures describe THIS run set,
they are not a claim about production quality. Failures are listed case by case."""
from __future__ import annotations

import json
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .bootstrap import open_store
from .config import Settings
from .evidence import Evidence
from .pipeline import FeedbackPipeline
from .schemas import URGENCY_ORDER, Feedback, Urgency
from .store import Store
from .tools import KnowledgeBase
from .trace import load_trace

REQUIRED_EVENTS = {"intake", "classification_completed", "report_generated"}


def load_cases(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text("utf-8"))["cases"]


def _pct(sorted_vals: list[float], quantile: float) -> Optional[int]:
    if not sorted_vals:
        return None
    return int(sorted_vals[max(0, math.ceil(quantile * len(sorted_vals)) - 1)])


def score_run(case: dict, report, store: Store, trace_dir: str, caps: Settings) -> dict[str, Any]:
    label = case["label"]
    evidence = Evidence.from_dict(json.loads(store.get_report_row(report.report_id)["evidence_json"]))
    action_types = [action.type.value for action in report.suggested_actions]
    cited = (set(report.cited_guidelines) | set(report.cited_policies)
             | {basis_id for action in report.suggested_actions for basis_id in action.basis})
    events = load_trace(report.trace_id, trace_dir)
    kinds = {event["event"] for event in events}
    blocked = sum(1 for event in events if event["event"] == "grounding_validation" and event.get("issues")
                  and any("not present" in issue or "not found in this run" in issue for issue in event["issues"]))
    text = report.model_dump_json()
    counters = report.counters
    expected_guidelines, expected_policies = label.get("expected_guidelines", []), label.get("expected_policies", [])
    row = {
        "case": case["id"], "predicted": report.category.value, "expected": label["category"],
        "category_ok": report.category.value == label["category"],
        "urgency_ok": URGENCY_ORDER.index(report.urgency) >= URGENCY_ORDER.index(Urgency(label["min_urgency"])),
        "triage_flagged": report.needs_human_triage, "triage_expected": label["needs_triage"],
        "actions": action_types,
        "actions_ok": not (set(action_types) & set(label.get("forbidden_actions", [])))
                      and (not label.get("allowed_actions") or set(action_types) <= set(label["allowed_actions"])),
        "flags_ok": all(flag in report.flags for flag in label.get("must_flag", [])),
        "mention_ok": not any(phrase in text for phrase in label.get("must_not_mention", [])),
        "customer_ok": report.customer_context.record_found == label["customer_found"],
        "expected_guidelines": expected_guidelines,
           "gl_hits": [guideline_id for guideline_id in expected_guidelines if guideline_id in evidence.guidelines],
        "expected_policies": expected_policies,
           "pol_hits": [policy_id for policy_id in expected_policies if policy_id in evidence.policies],
        "stale_citation": [policy_id for policy_id in label.get("forbidden_citations", []) if policy_id in cited],
        "leaked_refs": sorted(cited - evidence.all_ids()),          # ids in the FINAL report not backed by tool results
        "blocked_refs_in_drafts": blocked,                     # bad ids the validator stopped before the report
        "all_three_sources": set(evidence.consulted) >= {"guidelines", "customer", "policies"},
        "degraded": report.report_generated_by == "degraded_template",
        "generated_by": report.report_generated_by,
        "within_caps": counters["llm_turns"] <= caps.max_llm_turns and counters["tool_calls"] <= caps.max_tool_calls
                       and counters["forced_by_system"] <= caps.max_forced_lookups,
        "step_limit": "step_limit_reached" in report.flags,
        "trace_complete": REQUIRED_EVENTS <= kinds and "tool_call" in kinds,
        "cost_usd": counters.get("cost_usd"), "usage_complete": counters.get("usage_complete"),
        "latency_ms": counters["elapsed_ms"], "prompt_tokens": counters["prompt_tokens"],
           "output_tokens": counters["output_tokens"],
        "llm_turns": counters["llm_turns"], "tool_calls": counters["tool_calls"],
           "forced": counters["forced_by_system"],
        "confidence_level": report.confidence_level, "tags": label.get("tags", []),
    }
    return row


def _error_row(case: dict, e: Exception) -> dict[str, Any]:
    """A run that raised: no report, so every label counts as missed (never silently dropped)."""
    label = case["label"]
    return {"case": case["id"], "error": f"{type(e).__name__}: {str(e)[:200]}", "predicted": None,
            "expected": label["category"], "category_ok": False, "urgency_ok": False, "actions_ok": False,
            "flags_ok": False, "mention_ok": False, "customer_ok": False, "triage_flagged": False,
            "triage_expected": label["needs_triage"], "expected_guidelines": label.get("expected_guidelines", []),
            "gl_hits": [], "expected_policies": label.get("expected_policies", []), "pol_hits": [],
            "tags": label.get("tags", []), "degraded": False}


def summarize(rows: list[dict], settings: Settings) -> dict[str, Any]:
    run_count = len(rows)
    scored_rows = [r for r in rows if "error" not in r]      # runs that produced a report
    # label-based metrics use ALL runs: a run that crashed counts as a miss, not as absent
    frac = lambda key, subset=None: f"{sum(1 for r in (subset if subset is not None else rows) if r[key])}/{len(subset if subset is not None else rows)}"
    frac_ok = lambda key: f"{sum(1 for r in scored_rows if r[key])}/{len(scored_rows)}"
    triage_expected_rows = [r for r in rows if r["triage_expected"]]
    triage_unexpected_rows = [r for r in rows if not r["triage_expected"]]
    expected_guideline_count = sum(len(r["expected_guidelines"]) for r in rows)
    expected_policy_count = sum(len(r["expected_policies"]) for r in rows)
    injection_rows = [r for r in rows if "injection" in r["tags"]]
    latencies = sorted(r["latency_ms"] for r in scored_rows)
    prompt_tokens = [r["prompt_tokens"] for r in scored_rows if r["prompt_tokens"] is not None]
    output_tokens = [r["output_tokens"] for r in scored_rows if r["output_tokens"] is not None]
    error_cases = [{"case": r["case"], "error": r["error"]} for r in rows if "error" in r]
    case_results = [{"case": r["case"], "predicted": None, "expected": r["expected"], "why": ["error"]}
                    for r in rows if "error" in r] + [{"case": r["case"], "why": [key for key in ("category_ok", "urgency_ok", "actions_ok", "flags_ok", "mention_ok",
                                                    "customer_ok") if not r[key]]
              + (["triage"] if r["triage_flagged"] != r["triage_expected"] else [])
              + (["retrieval_miss"] if len(r["gl_hits"]) < len(r["expected_guidelines"])
                 or len(r["pol_hits"]) < len(r["expected_policies"]) else [])
              + (["stale_citation"] if r["stale_citation"] else []) + (["degraded"] if r["degraded"] else []),
              "predicted": r["predicted"], "expected": r["expected"]} for r in scored_rows]
    costs = [r["cost_usd"] for r in scored_rows if isinstance(r.get("cost_usd"), (int, float))]
    return {
        "runs": run_count, "errors": run_count - len(scored_rows), "scored_runs": len(scored_rows), "error_cases": error_cases, "caps": {"max_llm_turns": settings.max_llm_turns, "max_tool_calls": settings.max_tool_calls},
        # a crashed run is not a completion
        "completed_normally": f"{sum(1 for r in scored_rows if not r['degraded'])}/{run_count}",
        "degraded_runs": sum(1 for r in scored_rows if r["degraded"]),
        "category_accuracy": frac("category_ok"), "urgency_at_least_label": frac("urgency_ok"),
        "triage_recall_on_ambiguous":
            f"{sum(1 for r in triage_expected_rows if r['triage_flagged'])}/{len(triage_expected_rows)}",
        "triage_false_positives":
            f"{sum(1 for r in triage_unexpected_rows if r['triage_flagged'])}/{len(triage_unexpected_rows)}",
        "actions_respect_labels": frac("actions_ok"),
        "retrieval_hit_rate": {
            "guidelines": f"{sum(len(r['gl_hits']) for r in scored_rows)}/{expected_guideline_count}",
            "policies": f"{sum(len(r['pol_hits']) for r in scored_rows)}/{expected_policy_count}"},
        "stale_policy_citations": sum(len(r["stale_citation"]) for r in scored_rows),
        "fabricated_refs_in_final_reports": sum(len(r["leaked_refs"]) for r in scored_rows),
        "fabricated_refs_blocked_in_drafts": sum(r["blocked_refs_in_drafts"] for r in scored_rows),
        "runs_with_all_three_sources": frac_ok("all_three_sources"),
        "runs_within_caps": frac_ok("within_caps"),
        "runs_hitting_step_limit": sum(1 for r in scored_rows if r["step_limit"]),
        "traces_complete": frac_ok("trace_complete"),
        "injection_cases_passed":
            f"{sum(1 for r in injection_rows if r['flags_ok'] and r['actions_ok'] and r['mention_ok'])}"
            f"/{len(injection_rows)}",
        "latency_ms": {"p50": _pct(latencies, 0.5), "p95": _pct(latencies, 0.95),
                       "note": "describes this run set only"},
        "tokens_per_run": {"prompt": round(sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else None,
                           "output": round(sum(output_tokens) / len(output_tokens)) if output_tokens else None},
        "avg_llm_turns": round(sum(r["llm_turns"] for r in scored_rows) / len(scored_rows), 2) if scored_rows else None,
        "avg_tool_calls": round(sum(r["tool_calls"] for r in scored_rows) / len(scored_rows),
                                2) if scored_rows else None,
        "cost_usd": round(sum(costs), 6) if costs else "unknown (set PRICE_PER_MTOK_IN/OUT to estimate)",
        "usage_complete_runs": f"{sum(1 for r in scored_rows if r.get('usage_complete'))}/{len(scored_rows)}",
        "case_results": case_results,
    }


def run_eval(settings: Settings, cases: list[dict], provider_factory: Callable[[dict], Any], repeats: int = 1,
             workers: int = 4, sweep: Optional[list[int]] = None, trace_root: Optional[str] = None,
             progress: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    knowledge_base = KnowledgeBase.load(settings.data_dir)
    with tempfile.TemporaryDirectory() as temp_dir:
        base_settings = deepcopy(settings)
        base_settings.database_path = f"{temp_dir}/eval.db"
        store = open_store(base_settings)
        results: dict[str, Any] = {"meta": {"started_at": datetime.now(timezone.utc).isoformat(), "cases": len(cases),
                                            "repeats": repeats, "workers": workers,
                                            "provider": settings.llm_provider, "model": settings.llm_model},
                                   "configs": []}
        for turns in (sweep or [settings.max_llm_turns]):
            turn_settings = deepcopy(base_settings)
            turn_settings.max_llm_turns = turns

            def one(item: tuple[dict, int], turn_settings=turn_settings) -> dict:
                case, repeat_no = item
                run_settings = deepcopy(turn_settings)
                run_dir = Path(trace_root or temp_dir) / f"turns{turns}" / f"{case['id']}.{repeat_no}"
                run_settings.trace_dir = str(run_dir)
                try:
                    feedback = Feedback(text=case["text"], customer_email=case["customer_email"],
                                        channel=case.get("channel", "email"), received_at=case["received_at"])
                    pipeline = FeedbackPipeline(run_settings, store, knowledge_base, provider_factory(case))
                    report = pipeline.process(feedback)
                    row = score_run(case, report, store, run_settings.trace_dir, turn_settings)
                except Exception as e:  # keep going: an eval run must report, not crash
                    row = _error_row(case, e)
                row["rep"] = repeat_no
                if progress:
                    progress(f"turns={turns} {case['id']}#{repeat_no} " + ("ERROR " + row["error"] if "error" in row
                             else f"{row['predicted']} by={row['generated_by']}"))
                return row

            items = [(case, repeat_no) for repeat_no in range(repeats) for case in cases]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(one, items))
            results["configs"].append({"max_llm_turns": turns, "summary": summarize(rows, turn_settings), "rows": rows})
        return results


def render_markdown(result: dict[str, Any]) -> str:
    meta = result["meta"]
    lines = [f"# Eval results ({meta['provider']} / {meta['model']})", "",
             f"{meta['cases']} labelled cases x {meta['repeats']} repeat(s), run {meta['started_at'][:19]}Z. Numbers describe this run set only: "
             "the sample is small, so no quality claim for production is made.", ""]
    for config in result["configs"]:
        summary = config["summary"]
        lines += [f"## max_llm_turns = {config['max_llm_turns']} "
                  f"(max_tool_calls = {summary['caps']['max_tool_calls']})", "",
                  "| Metric | Result |", "|---|---|"]
        for metric in ("runs", "errors", "scored_runs", "completed_normally", "degraded_runs", "category_accuracy", "urgency_at_least_label",
                       "triage_recall_on_ambiguous", "triage_false_positives", "actions_respect_labels", "stale_policy_citations",
                       "fabricated_refs_in_final_reports", "fabricated_refs_blocked_in_drafts", "runs_with_all_three_sources",
                       "runs_within_caps", "runs_hitting_step_limit", "traces_complete", "injection_cases_passed",
                       "avg_llm_turns", "avg_tool_calls"):
            lines.append(f"| {metric} | {summary[metric]} |")
        hit_rate, latency, tokens = summary["retrieval_hit_rate"], summary["latency_ms"], summary["tokens_per_run"]
        lines += [f"| retrieval_hit_rate | guidelines {hit_rate['guidelines']}, policies {hit_rate['policies']} |",
                  f"| latency_ms | p50 {latency['p50']}, p95 {latency['p95']} |",
                  f"| tokens_per_run | in {tokens['prompt']}, out {tokens['output']} |",
                  f"| cost_usd | {summary['cost_usd']} (usage complete in {summary['usage_complete_runs']} runs; "
                  "otherwise a lower bound) |", ""]
        if summary["errors"]:
            lines += [f"**{summary['errors']} of {summary['runs']} runs crashed and produced no report** "
                      "(counted as misses above, not dropped):"]
            lines += [f"- `{error_case['case']}`: {error_case['error']}" for error_case in summary["error_cases"]]
            lines += [""]
        missed = [case_result for case_result in summary["case_results"] if case_result["why"]]
        lines.append("Cases with a miss:" if missed else "No case missed a label in this configuration.")
        for case_result in missed:
            lines.append(f"- `{case_result['case']}`: {', '.join(case_result['why'])} "
                         f"(predicted `{case_result['predicted']}`, expected `{case_result['expected']}`)")
        lines.append("")
    return "\n".join(lines)

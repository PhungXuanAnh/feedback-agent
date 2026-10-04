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


def _pct(sorted_vals: list[float], q: float) -> Optional[int]:
    if not sorted_vals:
        return None
    return int(sorted_vals[max(0, math.ceil(q * len(sorted_vals)) - 1)])


def score_run(case: dict, rep, store: Store, trace_dir: str, caps: Settings) -> dict[str, Any]:
    lab = case["label"]
    ev = Evidence.from_dict(json.loads(store.get_report_row(rep.report_id)["evidence_json"]))
    acts = [a.type.value for a in rep.suggested_actions]
    cited = set(rep.cited_guidelines) | set(rep.cited_policies) | {b for a in rep.suggested_actions for b in a.basis}
    events = load_trace(rep.trace_id, trace_dir)
    kinds = {e["event"] for e in events}
    blocked = sum(1 for e in events if e["event"] == "grounding_validation" and e.get("issues")
                  and any("not present" in i or "not found in this run" in i for i in e["issues"]))
    text = rep.model_dump_json()
    c = rep.counters
    exp_gl, exp_pol = lab.get("expected_guidelines", []), lab.get("expected_policies", [])
    row = {
        "case": case["id"], "predicted": rep.category.value, "expected": lab["category"],
        "category_ok": rep.category.value == lab["category"],
        "urgency_ok": URGENCY_ORDER.index(rep.urgency) >= URGENCY_ORDER.index(Urgency(lab["min_urgency"])),
        "triage_flagged": rep.needs_human_triage, "triage_expected": lab["needs_triage"],
        "actions": acts,
        "actions_ok": not (set(acts) & set(lab.get("forbidden_actions", [])))
                      and (not lab.get("allowed_actions") or set(acts) <= set(lab["allowed_actions"])),
        "flags_ok": all(f in rep.flags for f in lab.get("must_flag", [])),
        "mention_ok": not any(m in text for m in lab.get("must_not_mention", [])),
        "customer_ok": rep.customer_context.record_found == lab["customer_found"],
        "expected_guidelines": exp_gl, "gl_hits": [g for g in exp_gl if g in ev.guidelines],
        "expected_policies": exp_pol, "pol_hits": [p for p in exp_pol if p in ev.policies],
        "stale_citation": [p for p in lab.get("forbidden_citations", []) if p in cited],
        "leaked_refs": sorted(cited - ev.all_ids()),          # ids in the FINAL report not backed by tool results
        "blocked_refs_in_drafts": blocked,                     # bad ids the validator stopped before the report
        "all_three_sources": set(ev.consulted) >= {"guidelines", "customer", "policies"},
        "degraded": rep.report_generated_by == "degraded_template",
        "generated_by": rep.report_generated_by,
        "within_caps": c["llm_turns"] <= caps.max_llm_turns and c["tool_calls"] <= caps.max_tool_calls
                       and c["forced_by_system"] <= caps.max_forced_lookups,
        "step_limit": "step_limit_reached" in rep.flags,
        "trace_complete": REQUIRED_EVENTS <= kinds and "tool_call" in kinds,
        "cost_usd": c.get("cost_usd"), "usage_complete": c.get("usage_complete"),
        "latency_ms": c["elapsed_ms"], "prompt_tokens": c["prompt_tokens"], "output_tokens": c["output_tokens"],
        "llm_turns": c["llm_turns"], "tool_calls": c["tool_calls"], "forced": c["forced_by_system"],
        "confidence_level": rep.confidence_level, "tags": lab.get("tags", []),
    }
    return row


def _error_row(case: dict, e: Exception) -> dict[str, Any]:
    """A run that raised: no report, so every label counts as missed (never silently dropped)."""
    lab = case["label"]
    return {"case": case["id"], "error": f"{type(e).__name__}: {str(e)[:200]}", "predicted": None,
            "expected": lab["category"], "category_ok": False, "urgency_ok": False, "actions_ok": False,
            "flags_ok": False, "mention_ok": False, "customer_ok": False, "triage_flagged": False,
            "triage_expected": lab["needs_triage"], "expected_guidelines": lab.get("expected_guidelines", []),
            "gl_hits": [], "expected_policies": lab.get("expected_policies", []), "pol_hits": [],
            "tags": lab.get("tags", []), "degraded": False}


def summarize(rows: list[dict], settings: Settings) -> dict[str, Any]:
    n = len(rows)
    ok = [r for r in rows if "error" not in r]      # runs that produced a report
    # label-based metrics use ALL runs: a run that crashed counts as a miss, not as absent
    frac = lambda k, rs=None: f"{sum(1 for r in (rs if rs is not None else rows) if r[k])}/{len(rs if rs is not None else rows)}"
    frac_ok = lambda k: f"{sum(1 for r in ok if r[k])}/{len(ok)}"
    tri_pos = [r for r in rows if r["triage_expected"]]
    tri_neg = [r for r in rows if not r["triage_expected"]]
    exp_gl = sum(len(r["expected_guidelines"]) for r in rows)
    exp_pol = sum(len(r["expected_policies"]) for r in rows)
    inj = [r for r in rows if "injection" in r["tags"]]
    lat = sorted(r["latency_ms"] for r in ok)
    toks_in = [r["prompt_tokens"] for r in ok if r["prompt_tokens"] is not None]
    toks_out = [r["output_tokens"] for r in ok if r["output_tokens"] is not None]
    errs = [{"case": r["case"], "error": r["error"]} for r in rows if "error" in r]
    fails = [{"case": r["case"], "predicted": None, "expected": r["expected"], "why": ["error"]}
             for r in rows if "error" in r] + [{"case": r["case"], "why": [k for k in ("category_ok", "urgency_ok", "actions_ok", "flags_ok", "mention_ok",
                                                    "customer_ok") if not r[k]]
              + (["triage"] if r["triage_flagged"] != r["triage_expected"] else [])
              + (["retrieval_miss"] if len(r["gl_hits"]) < len(r["expected_guidelines"])
                 or len(r["pol_hits"]) < len(r["expected_policies"]) else [])
              + (["stale_citation"] if r["stale_citation"] else []) + (["degraded"] if r["degraded"] else []),
              "predicted": r["predicted"], "expected": r["expected"]} for r in ok]
    costs = [r["cost_usd"] for r in ok if isinstance(r.get("cost_usd"), (int, float))]
    return {
        "runs": n, "errors": n - len(ok), "scored_runs": len(ok), "error_cases": errs, "caps": {"max_llm_turns": settings.max_llm_turns, "max_tool_calls": settings.max_tool_calls},
        "completed_normally": f"{sum(1 for r in ok if not r['degraded'])}/{n}",  # a crashed run is not a completion
        "degraded_runs": sum(1 for r in ok if r["degraded"]),
        "category_accuracy": frac("category_ok"), "urgency_at_least_label": frac("urgency_ok"),
        "triage_recall_on_ambiguous": f"{sum(1 for r in tri_pos if r['triage_flagged'])}/{len(tri_pos)}",
        "triage_false_positives": f"{sum(1 for r in tri_neg if r['triage_flagged'])}/{len(tri_neg)}",
        "actions_respect_labels": frac("actions_ok"),
        "retrieval_hit_rate": {"guidelines": f"{sum(len(r['gl_hits']) for r in ok)}/{exp_gl}",
                               "policies": f"{sum(len(r['pol_hits']) for r in ok)}/{exp_pol}"},
        "stale_policy_citations": sum(len(r["stale_citation"]) for r in ok),
        "fabricated_refs_in_final_reports": sum(len(r["leaked_refs"]) for r in ok),
        "fabricated_refs_blocked_in_drafts": sum(r["blocked_refs_in_drafts"] for r in ok),
        "runs_with_all_three_sources": frac_ok("all_three_sources"),
        "runs_within_caps": frac_ok("within_caps"), "runs_hitting_step_limit": sum(1 for r in ok if r["step_limit"]),
        "traces_complete": frac_ok("trace_complete"),
        "injection_cases_passed": f"{sum(1 for r in inj if r['flags_ok'] and r['actions_ok'] and r['mention_ok'])}/{len(inj)}",
        "latency_ms": {"p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95), "note": "describes this run set only"},
        "tokens_per_run": {"prompt": round(sum(toks_in) / len(toks_in)) if toks_in else None,
                           "output": round(sum(toks_out) / len(toks_out)) if toks_out else None},
        "avg_llm_turns": round(sum(r["llm_turns"] for r in ok) / len(ok), 2) if ok else None,
        "avg_tool_calls": round(sum(r["tool_calls"] for r in ok) / len(ok), 2) if ok else None,
        "cost_usd": round(sum(costs), 6) if costs else "unknown (set PRICE_PER_MTOK_IN/OUT to estimate)",
        "usage_complete_runs": f"{sum(1 for r in ok if r.get('usage_complete'))}/{len(ok)}",
        "case_results": fails,
    }


def run_eval(settings: Settings, cases: list[dict], provider_factory: Callable[[dict], Any], repeats: int = 1,
             workers: int = 4, sweep: Optional[list[int]] = None, trace_root: Optional[str] = None,
             progress: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    kb = KnowledgeBase.load(settings.data_dir)
    with tempfile.TemporaryDirectory() as tmp:
        base = deepcopy(settings)
        base.database_path = f"{tmp}/eval.db"
        store = open_store(base)
        out: dict[str, Any] = {"meta": {"started_at": datetime.now(timezone.utc).isoformat(), "cases": len(cases),
                                        "repeats": repeats, "workers": workers,
                                        "provider": settings.llm_provider, "model": settings.llm_model},
                               "configs": []}
        for turns in (sweep or [settings.max_llm_turns]):
            cfg = deepcopy(base)
            cfg.max_llm_turns = turns

            def one(item: tuple[dict, int], cfg=cfg) -> dict:
                case, rep_no = item
                s = deepcopy(cfg)
                s.trace_dir = str(Path(trace_root or tmp) / f"turns{turns}" / f"{case['id']}.{rep_no}")
                try:
                    fb = Feedback(text=case["text"], customer_email=case["customer_email"],
                                  channel=case.get("channel", "email"), received_at=case["received_at"])
                    rep = FeedbackPipeline(s, store, kb, provider_factory(case)).process(fb)
                    row = score_run(case, rep, store, s.trace_dir, cfg)
                except Exception as e:  # keep going: an eval run must report, not crash
                    row = _error_row(case, e)
                row["rep"] = rep_no
                if progress:
                    progress(f"turns={turns} {case['id']}#{rep_no} " + ("ERROR " + row["error"] if "error" in row
                             else f"{row['predicted']} by={row['generated_by']}"))
                return row

            items = [(c, r) for r in range(repeats) for c in cases]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(one, items))
            out["configs"].append({"max_llm_turns": turns, "summary": summarize(rows, cfg), "rows": rows})
        return out


def render_markdown(result: dict[str, Any]) -> str:
    m = result["meta"]
    lines = [f"# Eval results ({m['provider']} / {m['model']})", "",
             f"{m['cases']} labelled cases x {m['repeats']} repeat(s), run {m['started_at'][:19]}Z. Numbers describe this run set only: "
             "the sample is small, so no quality claim for production is made.", ""]
    for cfg in result["configs"]:
        s = cfg["summary"]
        lines += [f"## max_llm_turns = {cfg['max_llm_turns']} (max_tool_calls = {s['caps']['max_tool_calls']})", "",
                  "| Metric | Result |", "|---|---|"]
        for k in ("runs", "errors", "scored_runs", "completed_normally", "degraded_runs", "category_accuracy", "urgency_at_least_label",
                  "triage_recall_on_ambiguous", "triage_false_positives", "actions_respect_labels", "stale_policy_citations",
                  "fabricated_refs_in_final_reports", "fabricated_refs_blocked_in_drafts", "runs_with_all_three_sources",
                  "runs_within_caps", "runs_hitting_step_limit", "traces_complete", "injection_cases_passed",
                  "avg_llm_turns", "avg_tool_calls"):
            lines.append(f"| {k} | {s[k]} |")
        lines += [f"| retrieval_hit_rate | guidelines {s['retrieval_hit_rate']['guidelines']}, policies {s['retrieval_hit_rate']['policies']} |",
                  f"| latency_ms | p50 {s['latency_ms']['p50']}, p95 {s['latency_ms']['p95']} |",
                  f"| tokens_per_run | in {s['tokens_per_run']['prompt']}, out {s['tokens_per_run']['output']} |",
                  f"| cost_usd | {s['cost_usd']} (usage complete in {s['usage_complete_runs']} runs; otherwise a lower bound) |", ""]
        if s["errors"]:
            lines += [f"**{s['errors']} of {s['runs']} runs crashed and produced no report** (counted as misses above, not dropped):"]
            lines += [f"- `{e['case']}`: {e['error']}" for e in s["error_cases"]] + [""]
        bad = [c for c in s["case_results"] if c["why"]]
        lines.append("Cases with a miss:" if bad else "No case missed a label in this configuration.")
        lines += [f"- `{c['case']}`: {', '.join(c['why'])} (predicted `{c['predicted']}`, expected `{c['expected']}`)" for c in bad]
        lines.append("")
    return "\n".join(lines)

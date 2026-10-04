"""Commands for the review queue, samples and the one-command demo."""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

from .bootstrap import open_store
from .config import Settings
from .hitl import ReviewError, ReviewRequest, ReviewService, pending_summary


def cmd_reports(a, s: Settings) -> int:
    for r in open_store(s).list_reports(a.status):
        print(json.dumps(pending_summary(r)))
    return 0


def cmd_show(a, s: Settings) -> int:
    try:
        print(json.dumps(ReviewService(open_store(s), s).get(a.report_id), indent=2, ensure_ascii=False))
        return 0
    except ReviewError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def cmd_review(a, s: Settings) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    overrides = json.loads(Path(a.overrides).read_text("utf-8")) if a.overrides else None
    try:
        out = ReviewService(open_store(s), s).review(a.report_id, ReviewRequest(
            decision=a.decision, actor=a.actor, note=a.note or "", overrides=overrides))
    except ReviewError as e:
        print(f"error {e.status_code}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


def cmd_serve(a, s: Settings) -> int:
    import uvicorn
    from .api import create_app
    uvicorn.run(create_app(s), host=a.host, port=a.port)
    return 0


def cmd_samples(a, s: Settings) -> int:
    from .samples import run_samples
    modes = ["live", "scripted"] if a.mode == "both" else [a.mode]
    for m in run_samples(s, modes, a.only):
        ok = "ok " if m["expectations_met"] else "MISMATCH"
        print(f"{ok} {m['sample']:<24} {m['mode']:<9} by={m['report_generated_by']:<18} trace={m['trace_id']} "
              f"{m['unmet_expectations'] or ''}")
    return 0


def cmd_demo(a, s: Settings) -> int:
    """Offline end-to-end demo (no API key): run one sample, show the trace, then approve it as an officer."""
    from .samples import SAMPLES, load_manifest
    from .trace import load_trace, render_trace
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sample = next(x for x in load_manifest() if x["name"] == a.sample)
    with tempfile.TemporaryDirectory() as tmp:
        s.database_path, s.trace_dir = f"{tmp}/demo.db", f"{tmp}/traces"
        from .bootstrap import build_pipeline
        from .samples import _faults
        from .schemas import Feedback
        s.llm_provider, s.llm_retry_backoff_s = "scripted", 0.1
        pipe = build_pipeline(s, script=str(SAMPLES / "scripts" / f"{sample['name']}.json"),
                              faults=_faults((sample.get("faults") or {}).get("inject_tool_fault")))
        fb = Feedback(**json.loads((SAMPLES / "inputs" / f"{sample['name']}.json").read_text("utf-8")))
        print(f"=== INPUT ({sample['title']})\n{fb.text}\n")
        rep = pipe.process(fb)
        print("=== REPORT\n" + rep.model_dump_json(indent=2, exclude_none=True))
        print("\n=== TRACE\n" + render_trace(load_trace(rep.trace_id, s.trace_dir)))
        svc = ReviewService(pipe.store, s)
        print("\n=== REVIEW QUEUE (pending_review)")
        for r in pipe.store.list_reports("pending_review"):
            print(json.dumps(pending_summary(r)))
        print("\n=== OFFICER APPROVES")
        print(json.dumps(svc.review(rep.report_id, ReviewRequest(decision="approve", actor="demo.officer")), indent=2))
    return 0


def register(sub) -> None:
    r = sub.add_parser("reports", help="list reports in the review queue")
    r.add_argument("--status", default="pending_review")
    r.set_defaults(fn=cmd_reports)
    sh = sub.add_parser("show", help="show a report with its review audit")
    sh.add_argument("report_id")
    sh.set_defaults(fn=cmd_show)
    rv = sub.add_parser("review", help="approve | override | reject a report")
    rv.add_argument("report_id")
    rv.add_argument("decision", choices=["approve", "override", "reject"])
    rv.add_argument("--actor", required=True)
    rv.add_argument("--note")
    rv.add_argument("--overrides", help="JSON file with category/urgency/suggested_actions overrides")
    rv.set_defaults(fn=cmd_review)
    sv = sub.add_parser("serve", help="start the HTTP API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)
    sm = sub.add_parser("samples", help="(re)generate samples/<name>/ outputs")
    sm.add_argument("--mode", choices=["live", "scripted", "both"], default="scripted")
    sm.add_argument("--only")
    sm.set_defaults(fn=cmd_samples)
    d = sub.add_parser("demo", help="offline end-to-end demo, no API key needed")
    d.add_argument("--sample", default="01_enterprise_outage")
    d.set_defaults(fn=cmd_demo)
    register_eval(sub)


def cmd_eval(a, s: Settings) -> int:
    from datetime import datetime, timezone
    from .config import ROOT
    from .evaluation import load_cases, render_markdown, run_eval
    from .llm import get_provider
    if a.provider:
        s.llm_provider = a.provider
    for k in ("max_tool_calls",):
        if getattr(a, k) is not None:
            setattr(s, k, getattr(a, k))
    cases = load_cases(a.cases)
    if a.only:
        cases = [c for c in cases if c["id"] in a.only.split(",")]
    sweep = [int(x) for x in a.sweep.split(",")] if a.sweep else ([a.max_llm_turns] if a.max_llm_turns else None)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_eval(s, cases, lambda case: get_provider(s), repeats=a.repeats, workers=a.workers, sweep=sweep,
                      trace_root=str(out_dir / "traces"), progress=lambda m: print(m, file=sys.stderr))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"eval-{stamp}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), "utf-8")
    md = render_markdown(result)
    if a.markdown:
        Path(a.markdown).write_text(md + "\n", "utf-8")
    print(md)
    errors = sum(c["summary"]["errors"] for c in result["configs"])
    if errors:
        print(f"eval FAILED: {errors} run(s) crashed without a report", file=sys.stderr)
    return 1 if errors else 0  # usable as a CI gate


def register_eval(sub) -> None:
    from .config import ROOT
    e = sub.add_parser("eval", help="run the labelled eval cases (needs a live provider)")
    e.add_argument("--cases", default=str(ROOT / "eval" / "cases.json"))
    e.add_argument("--provider")
    e.add_argument("--only", help="comma separated case ids")
    e.add_argument("--repeats", type=int, default=1)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--max-llm-turns", type=int)
    e.add_argument("--max-tool-calls", type=int)
    e.add_argument("--sweep", help="comma separated max_llm_turns values, e.g. 2,3,4,6")
    e.add_argument("--out", default=str(ROOT / "eval" / "results"))
    e.add_argument("--markdown", help="also write the summary to this .md file")
    e.set_defaults(fn=cmd_eval)

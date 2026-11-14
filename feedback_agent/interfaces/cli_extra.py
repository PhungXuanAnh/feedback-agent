"""Commands for the review queue, samples and the one-command demo."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

from ..agent_core.hitl import ReviewError, ReviewRequest, ReviewService, report_summary
from ..bootstrap import open_database
from ..config import Settings


def cmd_reports(args, settings: Settings) -> int:
    """`reports`: print one summary line per report, optionally only --status."""
    for row in open_database(settings).list_reports(args.status):
        print(json.dumps(report_summary(row)))
    return 0


def cmd_show(args, settings: Settings) -> int:
    """`show`: print a report with its status and review audit."""
    try:
        print(
            json.dumps(
                ReviewService(open_database(settings), settings).get(args.report_id),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    except ReviewError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def cmd_review(args, settings: Settings) -> int:
    """`review`: record an officer's decision (overrides from a JSON file), print the result."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    overrides = json.loads(Path(args.overrides).read_text("utf-8")) if args.overrides else None
    try:
        review_result = ReviewService(open_database(settings), settings).review(
            args.report_id,
            ReviewRequest(
                decision=args.decision, actor=args.actor, note=args.note or "", overrides=overrides
            ),
        )
    except ReviewError as e:
        print(f"error {e.status_code}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(review_result, indent=2))
    return 0


def cmd_serve(args, settings: Settings) -> int:
    """`serve`: start the HTTP API with uvicorn."""
    import uvicorn
    from .api import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


def cmd_samples(args, settings: Settings) -> int:
    """`samples`: regenerate samples/<name>/ outputs and say whether expectations were met."""
    from ..ops.samples import run_samples

    modes = ["live", "scripted"] if args.mode == "both" else [args.mode]
    for meta in run_samples(settings, modes, args.only):
        ok = "ok " if meta["expectations_met"] else "MISMATCH"
        print(
            f"{ok} {meta['sample']:<24} {meta['mode']:<9} by={meta['report_generated_by']:<18} "
            f"trace={meta['trace_id']} {meta['unmet_expectations'] or ''}"
        )
    return 0


def cmd_demo(args, settings: Settings) -> int:
    """`demo`: offline run of one sample (no API key): report, trace, queue, then approval."""
    from ..ops.samples import SAMPLES, load_manifest
    from ..ops.trace import load_trace, render_trace

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for candidate in load_manifest():
        if candidate["name"] == args.sample:
            sample = candidate
            break
    else:
        raise ValueError(f"unknown sample: {args.sample}")
    with tempfile.TemporaryDirectory() as temp_dir:
        settings.database_path, settings.trace_dir = f"{temp_dir}/demo.db", f"{temp_dir}/traces"
        from ..bootstrap import build_pipeline
        from ..ops.samples import _fault_injector
        from ..schemas import Feedback

        settings.llm_provider, settings.llm_retry_backoff_s = "scripted", 0.1
        pipeline = build_pipeline(
            settings,
            script=str(SAMPLES / "scripts" / f"{sample['name']}.json"),
            fault_injector=_fault_injector((sample.get("faults") or {}).get("inject_tool_fault")),
        )
        feedback = Feedback(
            **json.loads((SAMPLES / "inputs" / f"{sample['name']}.json").read_text("utf-8"))
        )
        print(f"=== INPUT ({sample['title']})\n{feedback.text}\n")
        report = pipeline.process(feedback)
        print("=== REPORT\n" + report.model_dump_json(indent=2, exclude_none=True))
        print("\n=== TRACE\n" + render_trace(load_trace(report.trace_id, settings.trace_dir)))
        review_service = ReviewService(pipeline.database, settings)
        print("\n=== REVIEW QUEUE (pending_review)")
        for row in pipeline.database.list_reports("pending_review"):
            print(json.dumps(report_summary(row)))
        print("\n=== OFFICER APPROVES")
        approval = ReviewRequest(decision="approve", actor="demo.officer")
        print(json.dumps(review_service.review(report.report_id, approval), indent=2))
    return 0


def register_commands(subparsers) -> None:
    """Add the reports, show, review, serve, samples, demo and eval subcommands."""
    reports_parser = subparsers.add_parser("reports", help="list reports in the review queue")
    reports_parser.add_argument("--status", default="pending_review")
    reports_parser.set_defaults(fn=cmd_reports)
    show_parser = subparsers.add_parser("show", help="show a report with its review audit")
    show_parser.add_argument("report_id")
    show_parser.set_defaults(fn=cmd_show)
    review_parser = subparsers.add_parser("review", help="approve | override | reject a report")
    review_parser.add_argument("report_id")
    review_parser.add_argument("decision", choices=["approve", "override", "reject"])
    review_parser.add_argument("--actor", required=True)
    review_parser.add_argument("--note")
    review_parser.add_argument(
        "--overrides", help="JSON file with category/urgency/suggested_actions overrides"
    )
    review_parser.set_defaults(fn=cmd_review)
    serve_parser = subparsers.add_parser("serve", help="start the HTTP API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(fn=cmd_serve)
    samples_parser = subparsers.add_parser("samples", help="(re)generate samples/<name>/ outputs")
    samples_parser.add_argument("--mode", choices=["live", "scripted", "both"], default="scripted")
    samples_parser.add_argument("--only")
    samples_parser.set_defaults(fn=cmd_samples)
    demo_parser = subparsers.add_parser("demo", help="offline end-to-end demo, no API key needed")
    demo_parser.add_argument("--sample", default="01_enterprise_outage")
    demo_parser.set_defaults(fn=cmd_demo)
    register_eval(subparsers)


def cmd_eval(args, settings: Settings) -> int:
    """`eval`: run the labelled cases on a live provider; write JSON and Markdown results.

    --sweep 3,4,5 repeats everything per max_llm_turns value. Exit 1 if a run crashed (CI gate).
    """
    from datetime import datetime, timezone
    from ..config import ROOT
    from ..llm import get_provider
    from ..ops.evaluation import load_cases, render_markdown, run_eval

    if args.provider:
        settings.llm_provider = args.provider
    for setting_name in ("max_tool_calls",):
        if getattr(args, setting_name) is not None:
            setattr(settings, setting_name, getattr(args, setting_name))
    cases = load_cases(args.cases)
    if args.only:
        cases = [case for case in cases if case["id"] in args.only.split(",")]
    sweep = (
        [int(value) for value in args.sweep.split(",")]
        if args.sweep
        else ([args.max_llm_turns] if args.max_llm_turns else None)
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_eval(
        settings,
        cases,
        lambda case: get_provider(settings),
        repeats=args.repeats,
        workers=args.workers,
        sweep=sweep,
        trace_root=str(out_dir / "traces"),
        progress=lambda message: print(message, file=sys.stderr),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"eval-{stamp}.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False), "utf-8"
    )
    markdown = render_markdown(result)
    if args.markdown:
        Path(args.markdown).write_text(markdown + "\n", "utf-8")
    print(markdown)
    errors = sum(case["summary"]["errors"] for case in result["configs"])
    if errors:
        print(f"eval FAILED: {errors} run(s) crashed without a report", file=sys.stderr)
    return 1 if errors else 0  # usable as a CI gate


def register_eval(subparsers) -> None:
    """Add the `eval` subcommand and its options."""
    from ..config import ROOT

    eval_parser = subparsers.add_parser(
        "eval", help="run the labelled eval cases (needs a live provider)"
    )
    eval_parser.add_argument("--cases", default=str(ROOT / "eval" / "cases.json"))
    eval_parser.add_argument("--provider")
    eval_parser.add_argument("--only", help="comma separated case ids")
    eval_parser.add_argument("--repeats", type=int, default=1)
    eval_parser.add_argument("--workers", type=int, default=3)
    eval_parser.add_argument("--max-llm-turns", type=int)
    eval_parser.add_argument("--max-tool-calls", type=int)
    eval_parser.add_argument("--sweep", help="comma separated max_llm_turns values, e.g. 2,3,4,6")
    eval_parser.add_argument("--out", default=str(ROOT / "eval" / "results"))
    eval_parser.add_argument("--markdown", help="also write the summary to this .md file")
    eval_parser.set_defaults(fn=cmd_eval)

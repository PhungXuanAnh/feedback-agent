"""Command line: python -m feedback_agent <command>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .bootstrap import build_pipeline, open_store
from .config import Settings
from .llm import ConfigError, FaultInjectingProvider, get_provider
from .schemas import Feedback
from .store import StorageError
from .tools import FaultInjector
from .trace import load_trace, render_trace


def _fault_injector(specs: list[str]) -> FaultInjector:
    """--inject-tool-fault search_policies[:N]  (N transient failures, default 1; -1 = always)"""
    injector = FaultInjector()
    for spec in specs or []:
        name, _, count = spec.partition(":")
        injector.tool_failures[name] = int(count) if count else 1
    return injector


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    if args.provider:
        settings.llm_provider = args.provider
    for setting_name in ("max_llm_turns", "max_tool_calls"):
        if getattr(args, setting_name) is not None:
            setattr(settings, setting_name, getattr(args, setting_name))
    if args.trace_dir:
        settings.trace_dir = args.trace_dir
    if args.db:
        settings.database_path = args.db
    exit_code = 0
    for path in args.inputs:
        script = args.script
        if settings.llm_provider == "scripted" and not script:
            script = Path(path).parent.parent / "scripts" / Path(path).name
        try:
            pipeline = build_pipeline(
                settings,
                script=str(script) if script else None,
                fault_injector=_fault_injector(args.inject_tool_fault),
            )
            if args.inject_llm_fault:
                pipeline.provider = FaultInjectingProvider(pipeline.provider, args.inject_llm_fault)
            report = pipeline.process(Feedback(**json.loads(Path(path).read_text("utf-8"))))
        except (ConfigError, StorageError, FileNotFoundError, ValidationError) as e:
            print(f"error: {e}", file=sys.stderr)
            exit_code = 1
            continue
        print(report.model_dump_json(indent=2))
        if args.out:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{Path(path).stem}.report.json").write_text(
                report.model_dump_json(indent=2) + "\n", "utf-8"
            )
        if args.show_trace:
            print("\n" + render_trace(load_trace(report.trace_id, settings.trace_dir)))
    return exit_code


def cmd_trace(args: argparse.Namespace, settings: Settings) -> int:
    print(render_trace(load_trace(args.trace, settings.trace_dir)))
    return 0


def cmd_seed(args: argparse.Namespace, settings: Settings) -> int:
    store = open_store(settings)
    print("seeded:", store.seed_from_csv(Path(settings.data_dir) / "seed"))
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(prog="feedback_agent", description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    run_parser = subparsers.add_parser(
        "run", help="process feedback JSON file(s) through the pipeline"
    )
    run_parser.add_argument("inputs", nargs="+")
    run_parser.add_argument("--provider", help="gemini | scripted (default: LLM_PROVIDER)")
    run_parser.add_argument("--script", help="scripted provider: JSON file of turns")
    run_parser.add_argument("--max-llm-turns", type=int)
    run_parser.add_argument("--max-tool-calls", type=int)
    run_parser.add_argument(
        "--inject-llm-fault", type=int, metavar="N", help="every LLM call from the Nth on fails"
    )
    run_parser.add_argument("--inject-tool-fault", action="append", metavar="TOOL[:N]")
    run_parser.add_argument("--trace-dir")
    run_parser.add_argument("--db")
    run_parser.add_argument("--out", help="directory for <input>.report.json")
    run_parser.add_argument("--show-trace", action="store_true")
    run_parser.set_defaults(fn=cmd_run)
    trace_parser = subparsers.add_parser("trace", help="print a trace in readable form")
    trace_parser.add_argument("trace", help="trace id or .jsonl path")
    trace_parser.set_defaults(fn=cmd_trace)
    seed_parser = subparsers.add_parser(
        "seed", help="(re)load customers/tickets/invoices from data/seed/*.csv"
    )
    seed_parser.set_defaults(fn=cmd_seed)
    from . import cli_extra  # review / reports / eval / serve / demo

    cli_extra.register(subparsers)
    args = parser.parse_args(argv)
    return args.fn(args, settings)


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line: python -m feedback_agent <command>."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
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
    f = FaultInjector()
    for spec in specs or []:
        name, _, n = spec.partition(":")
        f.tool_failures[name] = int(n) if n else 1
    return f


def cmd_run(a: argparse.Namespace, s: Settings) -> int:
    if a.provider:
        s.llm_provider = a.provider
    for k in ("max_llm_turns", "max_tool_calls"):
        if getattr(a, k) is not None:
            setattr(s, k, getattr(a, k))
    if a.trace_dir:
        s.trace_dir = a.trace_dir
    if a.db:
        s.database_path = a.db
    rc = 0
    for path in a.inputs:
        script = a.script
        if s.llm_provider == "scripted" and not script:
            script = Path(path).parent.parent / "scripts" / Path(path).name
        try:
            pipe = build_pipeline(s, script=str(script) if script else None, faults=_fault_injector(a.inject_tool_fault))
            if a.inject_llm_fault:
                pipe.provider = FaultInjectingProvider(pipe.provider, a.inject_llm_fault)
            report = pipe.process(Feedback(**json.loads(Path(path).read_text("utf-8"))))
        except (ConfigError, StorageError, FileNotFoundError, ValidationError) as e:
            print(f"error: {e}", file=sys.stderr)
            rc = 1
            continue
        print(report.model_dump_json(indent=2))
        if a.out:
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{Path(path).stem}.report.json").write_text(report.model_dump_json(indent=2) + "\n", "utf-8")
        if a.show_trace:
            print("\n" + render_trace(load_trace(report.trace_id, s.trace_dir)))
    return rc


def cmd_ask(a: argparse.Namespace, s: Settings) -> int:
    """Process one feedback typed on the command line (same pipeline, same review queue as `run`)."""
    payload = {"text": a.text, "customer_email": a.email, "channel": a.channel}
    if a.received_at:
        payload["received_at"] = a.received_at
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ask.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        ns = argparse.Namespace(inputs=[str(path)], provider=a.provider, script=a.script, max_llm_turns=None,
                                max_tool_calls=None, inject_llm_fault=None, inject_tool_fault=None, trace_dir=None,
                                db=None, out=None, show_trace=a.show_trace)
        return cmd_run(ns, s)


def cmd_trace(a: argparse.Namespace, s: Settings) -> int:
    print(render_trace(load_trace(a.trace, s.trace_dir)))
    return 0


def cmd_seed(a: argparse.Namespace, s: Settings) -> int:
    store = open_store(s)
    print("seeded:", store.seed_from_csv(Path(s.data_dir) / "seed"))
    return 0


def main(argv: list[str] | None = None) -> int:
    s = Settings.from_env()
    p = argparse.ArgumentParser(prog="feedback_agent", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process feedback JSON file(s) through the pipeline")
    r.add_argument("inputs", nargs="+")
    r.add_argument("--provider", help="gemini | scripted (default: LLM_PROVIDER)")
    r.add_argument("--script", help="scripted provider: JSON file of turns")
    r.add_argument("--max-llm-turns", type=int)
    r.add_argument("--max-tool-calls", type=int)
    r.add_argument("--inject-llm-fault", type=int, metavar="N", help="every LLM call from the Nth on fails")
    r.add_argument("--inject-tool-fault", action="append", metavar="TOOL[:N]")
    r.add_argument("--trace-dir")
    r.add_argument("--db")
    r.add_argument("--out", help="directory for <input>.report.json")
    r.add_argument("--show-trace", action="store_true")
    r.set_defaults(fn=cmd_run)
    k = sub.add_parser("ask", help="process one feedback typed on the command line")
    k.add_argument("text", help="the customer's feedback text")
    k.add_argument("--email", required=True, help="customer email (metadata); try ops@northwind-analytics.example")
    k.add_argument("--channel", default="web_form", choices=["email", "web_form", "chat", "api"])
    k.add_argument("--received-at", help="ISO time, e.g. 2026-09-22T09:00:00Z (default: now; it fixes which policies are in force)")
    k.add_argument("--provider", help="gemini | scripted (default: LLM_PROVIDER)")
    k.add_argument("--script", help="scripted provider: JSON file of turns")
    k.add_argument("--show-trace", action="store_true")
    k.set_defaults(fn=cmd_ask)
    t = sub.add_parser("trace", help="print a trace in readable form")
    t.add_argument("trace", help="trace id or .jsonl path")
    t.set_defaults(fn=cmd_trace)
    sd = sub.add_parser("seed", help="(re)load customers/tickets/invoices from data/seed/*.csv")
    sd.set_defaults(fn=cmd_seed)
    from . import cli_extra  # review / reports / eval / serve / demo
    cli_extra.register(sub)
    args = p.parse_args(argv)
    return args.fn(args, s)


if __name__ == "__main__":
    raise SystemExit(main())

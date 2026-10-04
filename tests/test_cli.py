import json

from conftest import C, G, P, calls, classify_step, submit
from feedback_agent.cli import main


def test_ask_runs_one_typed_feedback_and_rejects_bad_input(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "ask.db"))
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    script = tmp_path / "script.json"
    script.write_text(json.dumps([classify_step(), calls(G, C, P), submit()]))
    args = ["ask", "I was charged twice on 2026-09-10, 1200 USD each time.", "--email", "billing@orchid-robotics.example",
            "--received-at", "2026-09-22T09:00:00Z", "--provider", "scripted", "--script", str(script)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert '"report_generated_by": "scripted"' in out and '"status": "pending_review"' in out
    assert main(["reports"]) == 0 and '"category": "billing_issue"' in capsys.readouterr().out   # same review queue
    assert main(["ask", "hi", "--email", "not-an-email", "--provider", "scripted", "--script", str(script)]) == 1

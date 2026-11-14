"""Versioned prompt files (prompts/<name>.md); the version id is written into traces and reports."""

from pathlib import Path

CLASSIFY_PROMPT = "classify.v1"
AGENT_PROMPT = "agent.v1"
_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Text of a versioned prompt, e.g. "agent.v1" -> prompts/agent.v1.md."""
    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")

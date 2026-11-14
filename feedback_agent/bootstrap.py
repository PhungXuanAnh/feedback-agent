"""Wiring shared by the CLI and the API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .agent_core.pipeline import FeedbackPipeline
from .agent_core.tools import FaultInjector, KnowledgeBase
from .config import ROOT, Settings
from .database import Database
from .llm import get_provider


def open_database(settings: Settings) -> Database:
    """Open the SQLite DB and load the CSV seed data on first use."""
    database = Database(settings.database_path)
    database.init_schema()
    if database.customer_by_id("C-1001") is None:
        database.seed_from_csv(Path(settings.data_dir) / "seed")
    return database


def build_pipeline(
    settings: Settings,
    script: Any = None,
    fault_injector: Optional[FaultInjector] = None,
    provider: Any = None,
) -> FeedbackPipeline:
    """FeedbackPipeline from settings: knowledge base files, LLM provider and SQLite database.

    `script` feeds the scripted provider; `provider` replaces the provider entirely (tests).
    """
    knowledge_base = KnowledgeBase.load(settings.data_dir)
    provider = provider or get_provider(settings, script=script)
    return FeedbackPipeline(
        settings, open_database(settings), knowledge_base, provider, fault_injector
    )


__all__ = ["ROOT", "build_pipeline", "open_database"]

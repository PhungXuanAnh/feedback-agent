"""Wiring shared by the CLI and the API."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .config import ROOT, Settings
from .llm import get_provider
from .pipeline import FeedbackPipeline
from .store import Store
from .tools import FaultInjector, KnowledgeBase


def open_store(settings: Settings) -> Store:
    """Open the SQLite DB and load the CSV seed data on first use."""
    store = Store(settings.database_path)
    store.init_schema()
    if store.customer_by_id("C-1001") is None:
        store.seed_from_csv(Path(settings.data_dir) / "seed")
    return store


def build_pipeline(settings: Settings, script: Any = None, faults: Optional[FaultInjector] = None,
                   provider: Any = None) -> FeedbackPipeline:
    kb = KnowledgeBase.load(settings.data_dir)
    provider = provider or get_provider(settings, script=script)
    return FeedbackPipeline(settings, open_store(settings), kb, provider, faults)


__all__ = ["ROOT", "build_pipeline", "open_store"]

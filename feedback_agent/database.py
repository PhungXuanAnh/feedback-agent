"""SQLite repository: seeded reference data (customers, tickets, invoices) plus the
feedback / report / review tables. All SQL lives here so another DB can replace it."""

from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers(id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT,
    tier TEXT NOT NULL, signup_date TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tickets(id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, subject TEXT,
    status TEXT, opened_at TEXT);
CREATE TABLE IF NOT EXISTS invoices(id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, amount REAL NOT NULL,
    date TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS feedback(id TEXT PRIMARY KEY, customer_email TEXT, channel TEXT, received_at TEXT,
    text TEXT, injection_suspected INTEGER, category TEXT, sentiment TEXT, urgency TEXT, confidence REAL,
    classified_by TEXT, prompt_version TEXT, trace_id TEXT);
CREATE TABLE IF NOT EXISTS reports(id TEXT PRIMARY KEY, feedback_id TEXT NOT NULL, status TEXT NOT NULL,
    payload_json TEXT NOT NULL, evidence_json TEXT NOT NULL, flags_json TEXT, confidence_level TEXT,
    created_at TEXT NOT NULL, executed_at TEXT);
CREATE TABLE IF NOT EXISTS reviews(id INTEGER PRIMARY KEY AUTOINCREMENT, report_id TEXT NOT NULL, actor TEXT NOT NULL,
    decision TEXT NOT NULL, overrides_json TEXT, note TEXT, at TEXT NOT NULL, machine_category TEXT,
    machine_urgency TEXT);
"""


class StorageError(RuntimeError):
    """The database is unavailable; callers must not claim a report was queued."""


class Database:
    def __init__(self, path: str):
        """Remember the DB path and create its folder (":memory:" works for tests)."""
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        """A connection for one block of work, then closed; sqlite3 errors become StorageError.

        Autocommit mode: writes that need several statements use explicit BEGIN/COMMIT.
        """
        try:
            connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            connection.row_factory = sqlite3.Row
        except sqlite3.Error as e:  # pragma: no cover - defensive
            raise StorageError(str(e)) from e
        try:
            yield connection
        except sqlite3.Error as e:
            raise StorageError(str(e)) from e
        finally:
            connection.close()

    # ------------------------------------------------------------------ seed
    def init_schema(self) -> None:
        """Create the tables if they do not exist yet."""
        with self.conn() as connection:
            connection.executescript(SCHEMA)

    def seed_from_csv(self, seed_dir: str | Path) -> dict[str, int]:
        """Idempotent: reloads the reference tables from CSV (feedback/report tables untouched)."""
        seed_dir = Path(seed_dir)
        counts: dict[str, int] = {}
        self.init_schema()
        with self.conn() as connection:
            connection.execute("BEGIN")
            for table in ("customers", "tickets", "invoices"):
                with open(seed_dir / f"{table}.csv", newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
                connection.execute(f"DELETE FROM {table}")
                if rows:
                    columns = list(rows[0].keys())
                    connection.executemany(
                        f"INSERT INTO {table}({','.join(columns)}) VALUES({','.join('?' * len(columns))})",
                        [tuple(row[column] for column in columns) for row in rows],
                    )
                counts[table] = len(rows)
            connection.execute("COMMIT")
        return counts

    # ------------------------------------------------------- reference reads
    def customer_by_email(self, email: str) -> Optional[dict]:
        """Customer row for this email (case-insensitive), or None."""
        with self.conn() as connection:
            row = connection.execute(
                "SELECT * FROM customers WHERE lower(email)=lower(?)", (email,)
            ).fetchone()
        return dict(row) if row else None

    def customer_by_id(self, customer_id: str) -> Optional[dict]:
        """Customer row for this id, or None."""
        with self.conn() as connection:
            row = connection.execute(
                "SELECT * FROM customers WHERE id=?", (customer_id,)
            ).fetchone()
        return dict(row) if row else None

    def tickets_for(self, customer_id: str) -> list[dict]:
        """The customer's tickets, newest first."""
        with self.conn() as connection:
            rows = connection.execute(
                "SELECT id AS ticket_id, subject, status, opened_at FROM tickets WHERE customer_id=? "
                "ORDER BY opened_at DESC",
                (customer_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def invoices_for(self, customer_id: str) -> list[dict]:
        """The customer's invoices, newest first."""
        with self.conn() as connection:
            rows = connection.execute(
                "SELECT id AS invoice_id, amount, date, status FROM invoices WHERE customer_id=? "
                "ORDER BY date DESC, id",
                (customer_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------ feedback / reports
    def save_feedback(
        self,
        feedback: Any,
        flags_injection: bool,
        classification: Any,
        prompt_version: str,
        trace_id: str,
    ) -> None:
        """Insert or replace the feedback row with its classification (audit and analytics)."""
        with self.conn() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO feedback VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    feedback.feedback_id,
                    feedback.customer_email,
                    feedback.channel.value,
                    feedback.received_at.isoformat(),
                    feedback.text,
                    int(flags_injection),
                    classification.category.value,
                    classification.sentiment.value,
                    classification.urgency.value,
                    classification.confidence,
                    classification.classified_by,
                    prompt_version,
                    trace_id,
                ),
            )

    def save_report(self, report: Any, evidence: dict) -> None:
        """Insert a new report together with the evidence it was built from."""
        with self.conn() as connection:
            connection.execute(
                "INSERT INTO reports(id,feedback_id,status,payload_json,evidence_json,flags_json,"
                "confidence_level,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    report.report_id,
                    report.feedback_id,
                    report.status,
                    report.model_dump_json(),
                    json.dumps(evidence),
                    json.dumps(report.flags),
                    report.confidence_level,
                    report.created_at,
                ),
            )

    def get_report_row(self, report_id: str) -> Optional[dict]:
        """Raw reports row (payload and evidence still JSON strings), or None."""
        with self.conn() as connection:
            row = connection.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        return dict(row) if row else None

    def list_reports(self, status: Optional[str] = None) -> list[dict]:
        """Report rows, oldest first, optionally of one status (e.g. "pending_review")."""
        query, args = "SELECT * FROM reports", ()
        if status:
            query, args = query + " WHERE status=?", (status,)
        with self.conn() as connection:
            rows = connection.execute(query + " ORDER BY created_at, id", args).fetchall()
        return [dict(row) for row in rows]

    def usage_since(self, day_start_iso: str) -> tuple[int, int]:
        """(reports created at/after the timestamp, prompt+output tokens they used) for daily caps."""
        with self.conn() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM reports WHERE created_at >= ?", (day_start_iso,)
            ).fetchall()
        tokens = 0
        for row in rows:
            counters = json.loads(row["payload_json"]).get("counters", {})
            tokens += (counters.get("prompt_tokens") or 0) + (counters.get("output_tokens") or 0)
        return len(rows), tokens

    def reviews_for(self, report_id: str) -> list[dict]:
        """Audit rows of a report in the order they were written."""
        with self.conn() as connection:
            rows = connection.execute(
                "SELECT * FROM reviews WHERE report_id=? ORDER BY id", (report_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    # ----------------------------------------------------------------- review
    def apply_review(self, report_id: str, new_status: str, review: dict) -> bool:
        """Conditional status change + audit row in ONE transaction. Returns False when the report
        was no longer pending_review (somebody else decided first)."""
        with self.conn() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE reports SET status=? WHERE id=? AND status='pending_review'",
                (new_status, report_id),
            )
            if cursor.rowcount != 1:
                connection.execute("ROLLBACK")
                return False
            connection.execute(
                "INSERT INTO reviews(report_id,actor,decision,overrides_json,note,at,machine_category,"
                "machine_urgency) VALUES(?,?,?,?,?,?,?,?)",
                (
                    report_id,
                    review["actor"],
                    review["decision"],
                    review.get("overrides_json"),
                    review.get("note", ""),
                    review["at"],
                    review["machine_category"],
                    review["machine_urgency"],
                ),
            )
            connection.execute("COMMIT")
            return True

    def mark_executed(self, report_id: str, at: str) -> bool:
        """Mark an approved/overridden report executed; False if it was not in that state."""
        with self.conn() as connection:
            cursor = connection.execute(
                "UPDATE reports SET status='executed', executed_at=? WHERE id=? "
                "AND status IN ('approved','overridden') AND executed_at IS NULL",
                (at, report_id),
            )
            return cursor.rowcount == 1

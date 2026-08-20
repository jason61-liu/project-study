from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .models import RunState


class IdempotencyConflict(ValueError):
    pass


class VersionConflict(RuntimeError):
    pass


class SQLiteStore:
    """Durable shared state. Every query includes tenant_id as an isolation key."""

    def __init__(self, path: str | Path = "data/research.db") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=10000")
            self._local.connection = connection
        return connection

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def _initialize(self) -> None:
        connection = sqlite3.connect(self.path)
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS runs (
              tenant_id TEXT NOT NULL,
              run_id TEXT NOT NULL,
              user_id TEXT NOT NULL,
              state_json TEXT NOT NULL,
              version INTEGER NOT NULL,
              updated_at REAL NOT NULL,
              PRIMARY KEY (tenant_id, run_id)
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
              tenant_id TEXT NOT NULL,
              run_id TEXT NOT NULL,
              version INTEGER NOT NULL,
              state_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              PRIMARY KEY (tenant_id, run_id, version)
            );
            CREATE TABLE IF NOT EXISTS idempotency (
              tenant_id TEXT NOT NULL,
              idem_key TEXT NOT NULL,
              request_hash TEXT NOT NULL,
              run_id TEXT NOT NULL,
              PRIMARY KEY (tenant_id, idem_key)
            );
            CREATE TABLE IF NOT EXISTS approvals (
              tenant_id TEXT NOT NULL,
              run_id TEXT NOT NULL,
              user_id TEXT NOT NULL,
              approved_at REAL NOT NULL,
              PRIMARY KEY (tenant_id, run_id)
            );
            CREATE TABLE IF NOT EXISTS spans (
              tenant_id TEXT NOT NULL,
              run_id TEXT NOT NULL,
              span_json TEXT NOT NULL,
              created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS spans_run_idx ON spans(tenant_id, run_id);
            """
        )
        connection.close()

    @staticmethod
    def request_hash(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def create_run_idempotent(self, state: RunState, key: str, request_hash: str) -> RunState:
        """Atomically reserve a business intent and create its initial checkpoint."""
        connection = self._connection()
        now = time.time()
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT request_hash, run_id FROM idempotency WHERE tenant_id=? AND idem_key=?",
                (state.tenant_id, key),
            ).fetchone()
            if row:
                if row["request_hash"] != request_hash:
                    raise IdempotencyConflict("same idempotency key used with different request")
                run = connection.execute(
                    "SELECT state_json FROM runs WHERE tenant_id=? AND run_id=?",
                    (state.tenant_id, row["run_id"]),
                ).fetchone()
                if not run:
                    raise RuntimeError("idempotency ledger points to a missing run")
                connection.execute("COMMIT")
                return RunState.from_dict(json.loads(run["state_json"]))
            data = json.dumps(state.to_dict(), ensure_ascii=False)
            connection.execute(
                "INSERT INTO idempotency VALUES (?, ?, ?, ?)",
                (state.tenant_id, key, request_hash, state.run_id),
            )
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (state.tenant_id, state.run_id, state.user_id, data, state.version, now),
            )
            connection.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
                (state.tenant_id, state.run_id, state.version, data, now),
            )
            connection.execute("COMMIT")
            return state
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def create_run(self, state: RunState) -> None:
        data = json.dumps(state.to_dict(), ensure_ascii=False)
        now = time.time()
        connection = self._connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (state.tenant_id, state.run_id, state.user_id, data, state.version, now),
            )
            connection.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
                (state.tenant_id, state.run_id, state.version, data, now),
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def save(self, state: RunState, expected_version: int) -> None:
        next_version = expected_version + 1
        state.version = next_version
        data = json.dumps(state.to_dict(), ensure_ascii=False)
        now = time.time()
        connection = self._connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = connection.execute(
                "UPDATE runs SET state_json=?, version=?, updated_at=? "
                "WHERE tenant_id=? AND run_id=? AND version=?",
                (data, next_version, now, state.tenant_id, state.run_id, expected_version),
            )
            if cursor.rowcount != 1:
                raise VersionConflict("stale run state")
            connection.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
                (state.tenant_id, state.run_id, next_version, data, now),
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            state.version = expected_version
            raise

    def load(self, tenant_id: str, run_id: str) -> RunState | None:
        row = self._connection().execute(
            "SELECT state_json FROM runs WHERE tenant_id=? AND run_id=?", (tenant_id, run_id)
        ).fetchone()
        return RunState.from_dict(json.loads(row["state_json"])) if row else None

    def approve(self, tenant_id: str, run_id: str, user_id: str) -> None:
        self._connection().execute(
            "INSERT OR REPLACE INTO approvals VALUES (?, ?, ?, ?)",
            (tenant_id, run_id, user_id, time.time()),
        )

    def is_approved(self, tenant_id: str, run_id: str) -> bool:
        row = self._connection().execute(
            "SELECT 1 FROM approvals WHERE tenant_id=? AND run_id=?", (tenant_id, run_id)
        ).fetchone()
        return row is not None

    def record_span(self, tenant_id: str, run_id: str, span: dict[str, Any]) -> None:
        self._connection().execute(
            "INSERT INTO spans VALUES (?, ?, ?, ?)",
            (tenant_id, run_id, json.dumps(span, ensure_ascii=False), time.time()),
        )

    def spans(self, tenant_id: str, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection().execute(
            "SELECT span_json FROM spans WHERE tenant_id=? AND run_id=? ORDER BY created_at",
            (tenant_id, run_id),
        ).fetchall()
        return [json.loads(row["span_json"]) for row in rows]

    def metrics(self) -> dict[str, float]:
        connection = self._connection()
        total = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        completed = connection.execute(
            "SELECT COUNT(*) FROM runs WHERE json_extract(state_json, '$.status')='completed'"
        ).fetchone()[0]
        cost = connection.execute(
            "SELECT COALESCE(SUM(json_extract(state_json, '$.cost_usd')), 0) FROM runs"
        ).fetchone()[0]
        return {"runs_total": float(total), "runs_completed": float(completed), "cost_usd_total": float(cost)}

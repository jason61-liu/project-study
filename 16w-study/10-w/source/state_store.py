"""SQLite-backed queue, lease state machine, checkpoints and idempotency records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any
from uuid import uuid4

from versioning import VersionManifest


JSON = dict[str, Any]


class StaleLease(RuntimeError):
    pass


class IdempotencyConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    message_id: str
    tenant_id: str
    input: JSON
    versions: VersionManifest
    state: str
    stage: str
    attempt: int
    lease_epoch: int
    lease_owner: str | None
    lease_expires_at: float | None
    deadline_epoch: float
    checkpoint: JSON
    budget: JSON
    result: JSON | None
    last_error: str | None


@dataclass(frozen=True)
class EffectRecord:
    idempotency_key: str
    tenant_id: str
    request_hash: str
    status: str
    result: JSON | None
    execution_count: int


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    versions_json TEXT NOT NULL,
                    version_fingerprint TEXT NOT NULL,
                    state TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    lease_epoch INTEGER NOT NULL DEFAULT 0,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    not_before REAL NOT NULL,
                    deadline_epoch REAL NOT NULL,
                    checkpoint_json TEXT NOT NULL,
                    budget_json TEXT NOT NULL,
                    result_json TEXT,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE (tenant_id, message_id)
                );

                CREATE TABLE IF NOT EXISTS effects (
                    idempotency_key TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    execution_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS tasks_claim_idx
                    ON tasks(state, not_before, lease_expires_at);
                """
            )

    def enqueue(
        self,
        *,
        message_id: str,
        tenant_id: str,
        input_data: JSON,
        versions: VersionManifest,
        deadline_epoch: float,
        budget: JSON,
    ) -> tuple[str, bool]:
        now = time.time()
        task_id = f"task_{uuid4().hex[:20]}"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                """
                SELECT task_id, input_json, version_fingerprint
                FROM tasks WHERE tenant_id=? AND message_id=?
                """,
                (tenant_id, message_id),
            ).fetchone()
            if existing:
                if (
                    existing["input_json"] != _dump(input_data)
                    or existing["version_fingerprint"] != versions.fingerprint
                ):
                    db.rollback()
                    raise IdempotencyConflict(
                        "message_id reused with different input or version combination"
                    )
                db.commit()
                return str(existing["task_id"]), False
            db.execute(
                """
                INSERT INTO tasks(
                    task_id, message_id, tenant_id, input_json, versions_json,
                    version_fingerprint, state, stage, not_before, deadline_epoch,
                    checkpoint_json, budget_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'READY', 'MODEL', ?, ?, '{}', ?, ?, ?)
                """,
                (
                    task_id,
                    message_id,
                    tenant_id,
                    _dump(input_data),
                    _dump(versions.as_dict()),
                    versions.fingerprint,
                    now,
                    deadline_epoch,
                    _dump(budget),
                    now,
                    now,
                ),
            )
            db.commit()
        return task_id, True

    def claim(self, worker_id: str, *, lease_s: float) -> TaskRecord | None:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """
                SELECT task_id FROM tasks
                WHERE deadline_epoch > ? AND (
                    (state IN ('READY', 'RETRY') AND not_before <= ?)
                    OR (state = 'RUNNING' AND lease_expires_at < ?)
                )
                ORDER BY created_at, task_id
                LIMIT 1
                """,
                (now, now, now),
            ).fetchone()
            if row is None:
                db.commit()
                return None
            db.execute(
                """
                UPDATE tasks
                SET state='RUNNING', attempt=attempt+1, lease_epoch=lease_epoch+1,
                    lease_owner=?, lease_expires_at=?, updated_at=?
                WHERE task_id=?
                """,
                (worker_id, now + lease_s, now, row["task_id"]),
            )
            claimed = db.execute("SELECT * FROM tasks WHERE task_id=?", (row["task_id"],)).fetchone()
            db.commit()
        return self._task(claimed)

    def heartbeat(self, task_id: str, *, owner: str, epoch: int, lease_s: float) -> None:
        now = time.time()
        with self._connect() as db:
            changed = db.execute(
                """
                UPDATE tasks SET lease_expires_at=?, updated_at=?
                WHERE task_id=? AND state='RUNNING' AND lease_owner=? AND lease_epoch=?
                """,
                (now + lease_s, now, task_id, owner, epoch),
            ).rowcount
        if changed != 1:
            raise StaleLease("heartbeat rejected for stale worker")

    def checkpoint(
        self,
        task_id: str,
        *,
        owner: str,
        epoch: int,
        stage: str,
        checkpoint: JSON,
        budget: JSON,
    ) -> None:
        with self._connect() as db:
            changed = db.execute(
                """
                UPDATE tasks
                SET stage=?, checkpoint_json=?, budget_json=?, updated_at=?
                WHERE task_id=? AND state='RUNNING' AND lease_owner=? AND lease_epoch=?
                """,
                (
                    stage,
                    _dump(checkpoint),
                    _dump(budget),
                    time.time(),
                    task_id,
                    owner,
                    epoch,
                ),
            ).rowcount
        if changed != 1:
            raise StaleLease("checkpoint rejected for stale worker")

    def retry(self, task_id: str, *, owner: str, epoch: int, delay_s: float, error: str) -> None:
        now = time.time()
        with self._connect() as db:
            changed = db.execute(
                """
                UPDATE tasks
                SET state='RETRY', not_before=?, lease_owner=NULL, lease_expires_at=NULL,
                    last_error=?, updated_at=?
                WHERE task_id=? AND state='RUNNING' AND lease_owner=? AND lease_epoch=?
                """,
                (now + max(0.0, delay_s), error, now, task_id, owner, epoch),
            ).rowcount
        if changed != 1:
            raise StaleLease("retry rejected for stale worker")

    def complete(self, task_id: str, *, owner: str, epoch: int, result: JSON, budget: JSON) -> None:
        with self._connect() as db:
            changed = db.execute(
                """
                UPDATE tasks
                SET state='SUCCEEDED', stage='DONE', result_json=?, budget_json=?,
                    lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                WHERE task_id=? AND state='RUNNING' AND lease_owner=? AND lease_epoch=?
                """,
                (_dump(result), _dump(budget), time.time(), task_id, owner, epoch),
            ).rowcount
        if changed != 1:
            raise StaleLease("completion rejected for stale worker")

    def fail(self, task_id: str, *, owner: str, epoch: int, error: str) -> None:
        with self._connect() as db:
            changed = db.execute(
                """
                UPDATE tasks
                SET state='FAILED', last_error=?, lease_owner=NULL, lease_expires_at=NULL,
                    updated_at=?
                WHERE task_id=? AND state='RUNNING' AND lease_owner=? AND lease_epoch=?
                """,
                (error, time.time(), task_id, owner, epoch),
            ).rowcount
        if changed != 1:
            raise StaleLease("failure rejected for stale worker")

    def get_task(self, task_id: str) -> TaskRecord:
        with self._connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return self._task(row)

    def all_tasks(self) -> list[TaskRecord]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM tasks ORDER BY created_at, task_id").fetchall()
        return [self._task(row) for row in rows]

    def compact(self) -> None:
        """Merge WAL state into the database for a portable demo artifact."""

        with self._connect() as db:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("VACUUM")

    def begin_effect(self, *, key: str, tenant_id: str, request: JSON) -> EffectRecord:
        request_hash = _hash_request(tenant_id, request)
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM effects WHERE idempotency_key=?", (key,)).fetchone()
            if row is None:
                db.execute(
                    """
                    INSERT INTO effects(
                        idempotency_key, tenant_id, request_hash, status, created_at, updated_at
                    ) VALUES (?, ?, ?, 'PENDING', ?, ?)
                    """,
                    (key, tenant_id, request_hash, now, now),
                )
                row = db.execute("SELECT * FROM effects WHERE idempotency_key=?", (key,)).fetchone()
            elif row["tenant_id"] != tenant_id or row["request_hash"] != request_hash:
                db.rollback()
                raise IdempotencyConflict("idempotency key reused with different tenant or arguments")
            db.commit()
        return self._effect(row)

    def commit_effect(self, *, key: str, result: JSON) -> EffectRecord:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM effects WHERE idempotency_key=?", (key,)).fetchone()
            if row is None:
                db.rollback()
                raise KeyError(key)
            if row["status"] == "PENDING":
                db.execute(
                    """
                    UPDATE effects
                    SET status='SUCCEEDED', result_json=?, execution_count=execution_count+1,
                        updated_at=?
                    WHERE idempotency_key=? AND status='PENDING'
                    """,
                    (_dump(result), time.time(), key),
                )
            row = db.execute("SELECT * FROM effects WHERE idempotency_key=?", (key,)).fetchone()
            db.commit()
        return self._effect(row)

    def get_effect(self, key: str, *, tenant_id: str) -> EffectRecord | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM effects WHERE idempotency_key=? AND tenant_id=?",
                (key, tenant_id),
            ).fetchone()
        return None if row is None else self._effect(row)

    @staticmethod
    def _task(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=str(row["task_id"]),
            message_id=str(row["message_id"]),
            tenant_id=str(row["tenant_id"]),
            input=json.loads(row["input_json"]),
            versions=VersionManifest(**json.loads(row["versions_json"])),
            state=str(row["state"]),
            stage=str(row["stage"]),
            attempt=int(row["attempt"]),
            lease_epoch=int(row["lease_epoch"]),
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            deadline_epoch=float(row["deadline_epoch"]),
            checkpoint=json.loads(row["checkpoint_json"]),
            budget=json.loads(row["budget_json"]),
            result=None if row["result_json"] is None else json.loads(row["result_json"]),
            last_error=row["last_error"],
        )

    @staticmethod
    def _effect(row: sqlite3.Row) -> EffectRecord:
        return EffectRecord(
            idempotency_key=str(row["idempotency_key"]),
            tenant_id=str(row["tenant_id"]),
            request_hash=str(row["request_hash"]),
            status=str(row["status"]),
            result=None if row["result_json"] is None else json.loads(row["result_json"]),
            execution_count=int(row["execution_count"]),
        )


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_request(tenant_id: str, request: JSON) -> str:
    return hashlib.sha256(f"{tenant_id}:".encode() + _dump(request).encode()).hexdigest()

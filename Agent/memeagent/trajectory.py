from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid
from typing import Any


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _hash_payload(value: Any) -> str:
    raw = _json_dumps(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrajectoryRun:
    run_id: str
    input_key: str
    workflow_kind: str
    status: str
    topic: str
    input_mode: str
    input_json: dict[str, Any]
    output_json: dict[str, Any]
    error: str
    started_at: float
    finished_at: float | None


@dataclass(frozen=True)
class TrajectoryEvent:
    run_id: str
    step_index: int
    stage: str
    name: str
    payload: dict[str, Any]
    created_at: float


class MemeTrajectoryCache:
    """Persistent run trajectory cache for end-to-end MemeAgent workflows."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_trajectories (
                        run_id TEXT PRIMARY KEY,
                        input_key TEXT NOT NULL,
                        workflow_kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        input_mode TEXT NOT NULL,
                        input_json TEXT NOT NULL,
                        output_json TEXT NOT NULL,
                        error TEXT NOT NULL,
                        started_at REAL NOT NULL,
                        finished_at REAL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_trajectories_input_key "
                    "ON agent_trajectories(input_key, started_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_trajectories_status "
                    "ON agent_trajectories(status, started_at)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_trajectory_events (
                        run_id TEXT NOT NULL,
                        step_index INTEGER NOT NULL,
                        stage TEXT NOT NULL,
                        name TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        PRIMARY KEY(run_id, step_index),
                        FOREIGN KEY(run_id) REFERENCES agent_trajectories(run_id)
                            ON DELETE CASCADE
                    )
                    """
                )

    def start_run(
        self,
        *,
        workflow_kind: str,
        topic: str,
        context: str,
        image_paths: list[str],
        image_urls: list[str],
        input_mode: str = "",
        options: dict[str, Any] | None = None,
    ) -> str:
        input_payload = {
            "topic": topic,
            "context": context,
            "image_paths": image_paths,
            "image_urls": image_urls,
            "input_mode": input_mode,
            "options": options or {},
        }
        run_id = uuid.uuid4().hex
        now = time.time()
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO agent_trajectories(
                        run_id,
                        input_key,
                        workflow_kind,
                        status,
                        topic,
                        input_mode,
                        input_json,
                        output_json,
                        error,
                        started_at,
                        finished_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        _hash_payload(input_payload),
                        workflow_kind,
                        "running",
                        topic,
                        input_mode,
                        _json_dumps(input_payload),
                        "{}",
                        "",
                        now,
                        None,
                    ),
                )
                self._insert_event(
                    conn,
                    run_id=run_id,
                    stage="input",
                    name="run_started",
                    payload=input_payload,
                )
        return run_id

    def record_event(
        self,
        run_id: str,
        *,
        stage: str,
        name: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with closing(self._connect()) as conn:
            with conn:
                self._insert_event(
                    conn,
                    run_id=run_id,
                    stage=stage,
                    name=name,
                    payload=payload or {},
                )

    def finish_run(
        self,
        run_id: str,
        *,
        output: dict[str, Any] | None = None,
        status: str = "completed",
    ) -> None:
        now = time.time()
        output_payload = output or {}
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    UPDATE agent_trajectories
                    SET status = ?,
                        output_json = ?,
                        error = '',
                        finished_at = ?
                    WHERE run_id = ?
                    """,
                    (status, _json_dumps(output_payload), now, run_id),
                )
                self._insert_event(
                    conn,
                    run_id=run_id,
                    stage="output",
                    name="run_finished",
                    payload={"status": status, "output": output_payload},
                )

    def fail_run(
        self,
        run_id: str,
        *,
        error: str,
        output: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        output_payload = output or {}
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    UPDATE agent_trajectories
                    SET status = 'failed',
                        output_json = ?,
                        error = ?,
                        finished_at = ?
                    WHERE run_id = ?
                    """,
                    (_json_dumps(output_payload), error, now, run_id),
                )
                self._insert_event(
                    conn,
                    run_id=run_id,
                    stage="error",
                    name="run_failed",
                    payload={"error": error, "output": output_payload},
                )

    def get_run(self, run_id: str) -> TrajectoryRun | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT run_id, input_key, workflow_kind, status, topic, input_mode,
                       input_json, output_json, error, started_at, finished_at
                FROM agent_trajectories
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_run(row)

    def list_runs(self, limit: int = 20) -> list[TrajectoryRun]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT run_id, input_key, workflow_kind, status, topic, input_mode,
                       input_json, output_json, error, started_at, finished_at
                FROM agent_trajectories
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def list_events(self, run_id: str) -> list[TrajectoryEvent]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT run_id, step_index, stage, name, payload_json, created_at
                FROM agent_trajectory_events
                WHERE run_id = ?
                ORDER BY step_index ASC
                """,
                (run_id,),
            ).fetchall()
        return [
            TrajectoryEvent(
                run_id=row[0],
                step_index=int(row[1]),
                stage=row[2],
                name=row[3],
                payload=_json_loads(row[4]),
                created_at=float(row[5]),
            )
            for row in rows
        ]

    def _insert_event(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        stage: str,
        name: str,
        payload: dict[str, Any],
    ) -> None:
        row = conn.execute(
            "SELECT COALESCE(MAX(step_index), -1) + 1 FROM agent_trajectory_events "
            "WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        step_index = int(row[0] if row else 0)
        conn.execute(
            """
            INSERT INTO agent_trajectory_events(
                run_id,
                step_index,
                stage,
                name,
                payload_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                step_index,
                stage,
                name,
                _json_dumps(payload),
                time.time(),
            ),
        )

    def _row_to_run(self, row: tuple[Any, ...]) -> TrajectoryRun:
        return TrajectoryRun(
            run_id=row[0],
            input_key=row[1],
            workflow_kind=row[2],
            status=row[3],
            topic=row[4],
            input_mode=row[5],
            input_json=_json_loads(row[6]),
            output_json=_json_loads(row[7]),
            error=row[8],
            started_at=float(row[9]),
            finished_at=float(row[10]) if row[10] is not None else None,
        )

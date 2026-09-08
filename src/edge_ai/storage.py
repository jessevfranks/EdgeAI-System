"""SQLite history for experiments and their metrics."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


@dataclass(slots=True)
class ExperimentRecord:
    id: str
    status: str
    config: dict[str, Any]
    run_dir: str
    created_at: str
    error: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ExperimentRecord:
        return cls(
            id=row["id"],
            status=row["status"],
            config=json.loads(row["config_json"]),
            run_dir=row["run_dir"],
            created_at=row["created_at"],
            error=row["error"],
        )

    @property
    def action(self) -> str:
        return self.config["action"]

    @property
    def name(self) -> str:
        return self.config["name"]

    @property
    def scale(self) -> str:
        return self.config["scale"]

    @property
    def device(self) -> str:
        return self.config.get("device", "0")

    @property
    def best_checkpoint(self) -> str | None:
        checkpoints = list(Path(self.run_dir).rglob("best.pt"))
        if not checkpoints:
            return None
        return str(max(checkpoints, key=lambda path: path.stat().st_mtime).resolve())


class ExperimentStorage:
    def __init__(self, database_path: str | Path = "runs/experiments.sqlite3") -> None:
        self.path = Path(database_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    experiment_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (experiment_id, kind, step)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def create_experiment(self, config: ExperimentConfig) -> ExperimentRecord:
        experiment_id = uuid.uuid4().hex[:12]
        run_dir = self.path.parent / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?)",
                (
                    experiment_id,
                    "starting",
                    json.dumps(config.to_dict(), sort_keys=True),
                    str(run_dir),
                    datetime.now(UTC).isoformat(),
                    None,
                ),
            )
        return self.get(experiment_id)

    def get(self, experiment_id: str) -> ExperimentRecord:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown experiment: {experiment_id}")
        return ExperimentRecord.from_row(row)

    def list(self) -> list[ExperimentRecord]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
        return [ExperimentRecord.from_row(row) for row in rows]

    def set_status(
        self, experiment_id: str, status: str, error: str | None = None
    ) -> ExperimentRecord:
        with self._connect() as db:
            db.execute(
                "UPDATE experiments SET status = ?, error = ? WHERE id = ?",
                (status, error, experiment_id),
            )
        return self.get(experiment_id)

    def save_metric(self, experiment_id: str, kind: str, step: int, data: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO metrics VALUES (?, ?, ?, ?)
                ON CONFLICT(experiment_id, kind, step) DO UPDATE SET data_json = excluded.data_json""",
                (experiment_id, kind, step, json.dumps(data, sort_keys=True)),
            )

    def metrics(self, experiment_id: str, kind: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT step, data_json FROM metrics WHERE experiment_id = ? AND kind = ? ORDER BY step",
                (experiment_id, kind),
            ).fetchall()
        return [{"step": row["step"], **json.loads(row["data_json"])} for row in rows]

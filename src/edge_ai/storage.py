"""A small SQLite store for experiments and their metrics."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, dataset_fingerprint

FINISHED = {"completed", "failed", "stopped", "interrupted"}


def now() -> str:
    return datetime.now(UTC).isoformat()


def devices(value: str) -> set[str]:
    """Normalize `0`, `cuda:0`, and multi-GPU strings for queue checks."""
    value = value.lower().replace("cuda:", "")
    return {item.strip() for item in value.split(",") if item.strip()} or {"cpu"}


@dataclass(slots=True)
class ExperimentRecord:
    id: str
    action: str
    name: str
    scale: str
    status: str
    config: dict[str, Any]
    run_dir: str
    pid: int | None = None
    created_at: str = ""
    started_at: str | None = None
    ended_at: str | None = None
    error: str | None = None
    best_fitness: float | None = None
    current_iteration: int = 0
    total_iterations: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ExperimentRecord:
        return cls(
            id=row["id"],
            action=row["action"],
            name=row["name"],
            scale=row["scale"],
            status=row["status"],
            config=json.loads(row["config_json"]),
            run_dir=row["run_dir"],
            pid=row["pid"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            error=row["error"],
            best_fitness=row["best_fitness"],
            current_iteration=row["current_iteration"] or 0,
            total_iterations=row["total_iterations"],
        )

    @property
    def device(self) -> str:
        return str(self.config.get("device", "0"))

    @property
    def parent_id(self) -> str | None:
        return self.config.get("parent_id")

    @property
    def dataset_fingerprint(self) -> str:
        return self.config.get("dataset_fingerprint", "")

    @property
    def best_checkpoint(self) -> str | None:
        paths = list(Path(self.run_dir).rglob("best.pt"))
        if not paths:
            return None
        return str(max(paths, key=lambda path: path.stat().st_mtime).resolve())


class ExperimentStorage:
    """Store job state plus flexible JSON metric rows in two tables."""

    def __init__(self, database_path: str | Path = "runs/experiments.sqlite3") -> None:
        self.path = Path(database_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    name TEXT NOT NULL,
                    scale TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    pid INTEGER,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    error TEXT,
                    run_dir TEXT NOT NULL,
                    best_fitness REAL,
                    current_iteration INTEGER NOT NULL DEFAULT 0,
                    total_iterations INTEGER
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

    def create_experiment(self, config: ExperimentConfig) -> ExperimentRecord:
        config.experiment_id = config.experiment_id or uuid.uuid4().hex[:12]
        config.dataset_fingerprint = dataset_fingerprint(config.data)
        run_dir = config.run_root.expanduser().resolve() / config.experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        values = (
            config.experiment_id,
            config.action,
            config.name,
            config.scale.value,
            "queued",
            json.dumps(config.to_dict(), sort_keys=True),
            now(),
            str(run_dir),
            config.iterations if config.action == "tune" else None,
        )
        with self._connect() as db:
            db.execute(
                """INSERT INTO experiments
                (id, action, name, scale, status, config_json, created_at, run_dir, total_iterations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                values,
            )
        return self.get(config.experiment_id)

    def get(self, experiment_id: str) -> ExperimentRecord:
        with self._connect() as db:
            row = db.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown experiment: {experiment_id}")
        return ExperimentRecord.from_row(row)

    def list(
        self, statuses: list[str] | None = None, actions: list[str] | None = None
    ) -> list[ExperimentRecord]:
        clauses, values = [], []
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            values.extend(statuses)
        if actions:
            clauses.append(f"action IN ({','.join('?' for _ in actions)})")
            values.extend(actions)
        query = "SELECT * FROM experiments"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._connect() as db:
            rows = db.execute(query + " ORDER BY created_at DESC", values).fetchall()
        return [ExperimentRecord.from_row(row) for row in rows]

    def save_config(self, config: ExperimentConfig) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE experiments SET config_json = ? WHERE id = ?",
                (json.dumps(config.to_dict(), sort_keys=True), config.experiment_id),
            )

    def set_status(
        self, experiment_id: str, status: str, *, pid: int | None = None, error: str | None = None
    ) -> ExperimentRecord:
        started = now() if status == "running" else None
        ended = now() if status in FINISHED else None
        with self._connect() as db:
            db.execute(
                """UPDATE experiments SET status = ?, pid = ?, error = ?,
                started_at = COALESCE(?, started_at), ended_at = ? WHERE id = ?""",
                (status, pid, error, started, ended, experiment_id),
            )
        return self.get(experiment_id)

    def update_progress(
        self, experiment_id: str, *, step: int | None = None, fitness: float | None = None
    ) -> None:
        record = self.get(experiment_id)
        with self._connect() as db:
            db.execute(
                "UPDATE experiments SET current_iteration = ?, best_fitness = ? WHERE id = ?",
                (
                    step if step is not None else record.current_iteration,
                    fitness if fitness is not None else record.best_fitness,
                    experiment_id,
                ),
            )

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

    def claim_next(self) -> ExperimentRecord | None:
        """Reserve the oldest queued job whose device is currently free."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            queued = db.execute("SELECT * FROM experiments WHERE status = 'queued' ORDER BY created_at").fetchall()
            active = db.execute("SELECT config_json FROM experiments WHERE status = 'running'").fetchall()
            occupied = [devices(json.loads(row["config_json"])["device"]) for row in active]
            for row in queued:
                device = json.loads(row["config_json"])["device"]
                if not any(devices(device) & used for used in occupied):
                    db.execute("UPDATE experiments SET status = 'running' WHERE id = ?", (row["id"],))
                    db.commit()
                    return self.get(row["id"])
            db.commit()
        return None

    def clone(self, experiment_id: str) -> ExperimentRecord:
        source = self.get(experiment_id)
        config = ExperimentConfig.from_dict(source.config)
        config.experiment_id = None
        config.parent_id = source.id
        config.name += " copy"
        config.resume = False
        return self.create_experiment(config)

    def artifacts(self, experiment_id: str) -> list[dict[str, Any]]:
        root = Path(self.get(experiment_id).run_dir)
        return [
            {"name": path.name, "path": str(path), "size_bytes": path.stat().st_size}
            for path in root.rglob("*")
            if path.is_file()
        ]

    def export(self, records: list[ExperimentRecord] | None = None) -> list[dict[str, Any]]:
        rows = []
        for record in self.list() if records is None else records:
            row = {
                "id": record.id,
                "name": record.name,
                "action": record.action,
                "scale": record.scale,
                "status": record.status,
                "device": record.device,
                "dataset_fingerprint": record.dataset_fingerprint,
                "best_fitness": record.best_fitness,
                "created_at": record.created_at,
            }
            row.update({f"config.{key}": value for key, value in record.config.items() if not isinstance(value, dict)})
            kind = {"train": "epoch", "tune": "trial", "evaluate": "evaluation"}[record.action]
            metric_rows = self.metrics(record.id, kind)
            if metric_rows:
                scored = [row for row in metric_rows if isinstance(row.get("fitness"), (int, float))]
                best = max(scored, key=lambda row: row["fitness"]) if scored else metric_rows[-1]
                row.update({f"metric.{key}": value for key, value in best.items() if not isinstance(value, dict)})
                row.update(
                    {f"hyperparameter.{key}": value for key, value in best.get("hyperparameters", {}).items()}
                )
            summaries = self.metrics(record.id, "summary")
            if summaries:
                row.update({f"summary.{key}": value for key, value in summaries[0].items() if key != "step"})
            checkpoint = record.best_checkpoint
            if checkpoint:
                row["checkpoint_size_bytes"] = Path(checkpoint).stat().st_size
            rows.append(row)
        return rows

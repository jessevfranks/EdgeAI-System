import json
import sqlite3
from pathlib import Path

from model_development.config import ExperimentConfig
from model_development.storage import ExperimentStorage


def _config(tmp_path: Path, action: str = "train") -> ExperimentConfig:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("train: train\nval: val\nnames: [target]\n", encoding="utf-8")
    return ExperimentConfig(action=action, name="run", data=str(dataset))


def test_create_list_and_update_status(tmp_path: Path) -> None:
    storage = ExperimentStorage(tmp_path / "experiments.sqlite3")
    record = storage.create_experiment(_config(tmp_path))
    assert record.status == "starting"
    assert "experiment_id" not in record.config
    assert storage.list()[0].name == "run"
    assert storage.set_status(record.id, "completed").status == "completed"


def test_metric_upserts_are_idempotent(tmp_path: Path) -> None:
    storage = ExperimentStorage(tmp_path / "experiments.sqlite3")
    record = storage.create_experiment(_config(tmp_path))
    storage.save_metric(record.id, "epoch", 1, {"map50_95": 0.4})
    storage.save_metric(record.id, "epoch", 1, {"map50_95": 0.5})
    assert storage.metrics(record.id, "epoch") == [{"step": 1, "map50_95": 0.5}]


def test_best_checkpoint_finds_existing_run_weights(tmp_path: Path) -> None:
    storage = ExperimentStorage(tmp_path / "runs" / "experiments.sqlite3")
    record = storage.create_experiment(_config(tmp_path))
    checkpoint = Path(record.run_dir) / "train" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"best")
    assert record.best_checkpoint == str(checkpoint)


def test_legacy_database_preserves_history_and_accepts_new_runs(tmp_path: Path) -> None:
    path = tmp_path / "experiments.sqlite3"
    config = _config(tmp_path)
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE experiments (
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
            CREATE TABLE metrics (
                experiment_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                step INTEGER NOT NULL,
                data_json TEXT NOT NULL,
                PRIMARY KEY (experiment_id, kind, step)
            );
            """
        )
        db.execute(
            """INSERT INTO experiments
            (id, action, name, scale, status, config_json, created_at, error, run_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("old", "train", "run", "n", "failed", json.dumps(config.to_dict()),
             "2026-09-22", "training failed", str(tmp_path / "old")),
        )
        db.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?)",
            ("old", "epoch", 1, json.dumps({"map50_95": 0.5})),
        )

    storage = ExperimentStorage(path)
    old = storage.get("old")
    assert old.config == config.to_dict()
    assert old.status == "failed"
    assert old.error == "training failed"
    assert old.created_at == "2026-09-22"
    assert old.run_dir == str(tmp_path / "old")
    assert storage.metrics("old", "epoch") == [{"step": 1, "map50_95": 0.5}]
    new = storage.create_experiment(config)
    storage.save_metric(new.id, "epoch", 1, {"map50_95": 0.6})
    assert storage.set_status(new.id, "completed").status == "completed"
    reopened = ExperimentStorage(path)
    assert len(reopened.list()) == 2
    assert reopened.get("old") == old
    assert reopened.metrics(new.id, "epoch") == [{"step": 1, "map50_95": 0.6}]

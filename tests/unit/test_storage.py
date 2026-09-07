from pathlib import Path

from edge_ai.config import ExperimentConfig, ModelScale
from edge_ai.storage import ExperimentStorage


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text("train: train\nval: val\ntest: test\nnames: [object]\n", encoding="utf-8")
    return path


def _training(tmp_path: Path, *, name: str = "run", device: str = "0") -> ExperimentConfig:
    return ExperimentConfig(
        action="train",
        data=str(_dataset(tmp_path)),
        scale=ModelScale.NANO,
        name=name,
        device=device,
        database_path=tmp_path / "runs.sqlite3",
        run_root=tmp_path / "runs",
        artifact_root=tmp_path / "artifacts",
    )


def test_create_transition_and_clone(tmp_path: Path) -> None:
    storage = ExperimentStorage(tmp_path / "runs.sqlite3")
    record = storage.create_experiment(_training(tmp_path))
    assert record.status == "queued"
    storage.set_status(record.id, "running", pid=123)
    completed = storage.set_status(record.id, "completed")
    assert completed.status == "completed"
    clone = storage.clone(record.id)
    assert clone.parent_id == record.id
    assert clone.status == "queued"


def test_metric_upserts_are_idempotent(tmp_path: Path) -> None:
    storage = ExperimentStorage(tmp_path / "runs.sqlite3")
    record = storage.create_experiment(_training(tmp_path))
    storage.save_metric(record.id, "epoch", 1, {"map50_95": 0.4})
    storage.save_metric(record.id, "epoch", 1, {"map50_95": 0.5})
    assert storage.metrics(record.id, "epoch") == [{"step": 1, "map50_95": 0.5}]
    assert storage.export([record])[0]["metric.map50_95"] == 0.5


def test_trial_upsert_updates_best_summary(tmp_path: Path) -> None:
    storage = ExperimentStorage(tmp_path / "runs.sqlite3")
    config = _training(tmp_path)
    config.action = "tune"
    record = storage.create_experiment(config)
    storage.save_metric(record.id, "trial", 1, {"fitness": 0.4, "hyperparameters": {"lr0": 0.01}})
    storage.save_metric(record.id, "trial", 2, {"fitness": 0.6, "hyperparameters": {"lr0": 0.005}})
    storage.update_progress(record.id, step=2, fitness=0.6)
    refreshed = storage.get(record.id)
    assert refreshed.best_fitness == 0.6
    assert refreshed.current_iteration == 2
    assert storage.export([refreshed])[0]["hyperparameter.lr0"] == 0.005


def test_device_queue_selects_only_free_device(tmp_path: Path) -> None:
    storage = ExperimentStorage(tmp_path / "runs.sqlite3")
    first = storage.create_experiment(_training(tmp_path, name="first", device="0"))
    storage.set_status(first.id, "running", pid=123)
    storage.create_experiment(_training(tmp_path, name="blocked", device="0"))
    free = storage.create_experiment(_training(tmp_path, name="free", device="1"))
    assert storage.claim_next().id == free.id


def test_multi_gpu_device_conflicts_with_each_member(tmp_path: Path) -> None:
    storage = ExperimentStorage(tmp_path / "runs.sqlite3")
    first = storage.create_experiment(_training(tmp_path, name="multi", device="0,1"))
    storage.set_status(first.id, "running", pid=123)
    storage.create_experiment(_training(tmp_path, name="single", device="cuda:1"))
    assert storage.claim_next() is None

from pathlib import Path

from edge_ai.config import ExperimentConfig
from edge_ai.storage import ExperimentStorage


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

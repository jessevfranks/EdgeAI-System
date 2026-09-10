from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from model_development.config import ExperimentConfig
from model_development.experiments import evaluate_model, train_model, tune_model
from model_development.metrics import sync_metrics
from model_development.storage import ExperimentStorage
from model_development.worker import WORKFLOWS, run_worker


class FakeYOLO:
    calls: ClassVar[list[tuple[str, dict]]] = []

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint

    def train(self, **kwargs) -> None:
        self.calls.append(("train", kwargs))
        output = Path(kwargs["project"]) / kwargs["name"]
        (output / "weights").mkdir(parents=True, exist_ok=True)
        (output / "weights" / "best.pt").write_bytes(b"best")
        (output / "results.csv").write_text(
            "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
            "1,0.7,0.6,0.65,0.45\n",
            encoding="utf-8",
        )

    def tune(self, **kwargs) -> None:
        self.calls.append(("tune", kwargs))
        output = Path(kwargs["project"]) / kwargs["name"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "tune_results.ndjson").write_text(
            json.dumps(
                {
                    "iteration": 1,
                    "fitness": 0.5,
                    "hyperparameters": {"lr0": 0.005},
                    "datasets": {"dataset": {"metrics/mAP50-95(B)": 0.5}},
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def val(self, **kwargs):
        self.calls.append(("val", kwargs))
        return SimpleNamespace(
            results_dict={
                "metrics/precision(B)": 0.8,
                "metrics/recall(B)": 0.7,
                "metrics/mAP50(B)": 0.75,
                "metrics/mAP50-95(B)": 0.55,
            },
            speed={"inference": 4.2},
        )


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text("train: train\nval: val\ntest: test\nnames: [target]\n", encoding="utf-8")
    return path


def test_train_and_tune_keep_full_metric_history(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("edge_ai.experiments.yolo_class", lambda: FakeYOLO)
    storage = ExperimentStorage(tmp_path / "experiments.sqlite3")

    training = ExperimentConfig(action="train", name="train", data=str(_dataset(tmp_path)))
    train_record = storage.create_experiment(training)
    train_model(training, Path(train_record.run_dir))
    sync_metrics(storage, train_record)
    assert storage.metrics(train_record.id, "epoch")[0]["map50_95"] == 0.45

    tuning = ExperimentConfig(action="tune", name="tune", data=str(_dataset(tmp_path)), epochs=20)
    tune_record = storage.create_experiment(tuning)
    tune_model(tuning, Path(tune_record.run_dir))
    sync_metrics(storage, tune_record)
    assert storage.metrics(tune_record.id, "trial")[0]["hyperparameters"] == {"lr0": 0.005}
    assert FakeYOLO.calls[-1][1]["use_ray"] is False


def test_evaluate_returns_key_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("edge_ai.experiments.yolo_class", lambda: FakeYOLO)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"best")
    config = ExperimentConfig(
        action="evaluate",
        name="evaluate",
        data=str(_dataset(tmp_path)),
        weights=str(weights),
    )
    metrics = evaluate_model(config, tmp_path / "run")
    assert metrics["map50_95"] == 0.55
    assert metrics["speed_inference_ms"] == 4.2


def test_worker_records_success_and_failure(tmp_path: Path, monkeypatch) -> None:
    storage = ExperimentStorage(tmp_path / "experiments.sqlite3")
    successful = storage.create_experiment(
        ExperimentConfig(action="train", name="success", data=str(_dataset(tmp_path)))
    )
    monkeypatch.setitem(WORKFLOWS, "train", lambda config, run_dir: None)
    assert run_worker(storage.path, successful.id) == 0
    assert storage.get(successful.id).status == "completed"

    failed = storage.create_experiment(
        ExperimentConfig(action="train", name="failure", data=str(_dataset(tmp_path)))
    )

    def fail(config, run_dir):
        raise RuntimeError("training failed")

    monkeypatch.setitem(WORKFLOWS, "train", fail)
    assert run_worker(storage.path, failed.id) == 1
    assert storage.get(failed.id).error == "training failed"

    invalid = storage.create_experiment(
        ExperimentConfig(action="train", name="invalid", data=str(_dataset(tmp_path)))
    )
    Path(invalid.config["data"]).unlink()
    assert run_worker(storage.path, invalid.id) == 1
    assert storage.get(invalid.id).status == "failed"

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from edge_ai.config import ExperimentConfig, ModelScale
from edge_ai.experiments import (
    create_evaluation,
    evaluate_model,
    promote_best,
    train_model,
    tune_model,
)
from edge_ai.storage import ExperimentStorage


class FakeYOLO:
    calls: ClassVar[list[tuple[str, dict]]] = []

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self.model = SimpleNamespace(parameters=list)

    def train(self, **kwargs):
        self.calls.append(("train", kwargs))
        output = Path(kwargs["project"]) / kwargs["name"]
        (output / "weights").mkdir(parents=True, exist_ok=True)
        (output / "weights" / "best.pt").write_bytes(b"best")
        (output / "weights" / "last.pt").write_bytes(b"last")
        (output / "results.csv").write_text(
            "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
            "1,0.7,0.6,0.65,0.45\n",
            encoding="utf-8",
        )

    def tune(self, **kwargs):
        self.calls.append(("tune", kwargs))
        output = Path(kwargs["project"]) / kwargs["name"]
        (output / "weights").mkdir(parents=True, exist_ok=True)
        (output / "weights" / "best.pt").write_bytes(b"best")
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
    source = tmp_path / "dataset.yaml"
    source.write_text("train: train\nval: val\ntest: test\nnames: [target]\n", encoding="utf-8")
    return source


def _paths(tmp_path: Path) -> dict:
    return {
        "database_path": tmp_path / "runs.sqlite3",
        "run_root": tmp_path / "runs",
        "artifact_root": tmp_path / "artifacts",
    }


def test_train_and_tune_call_ultralytics_and_ingest_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("edge_ai.experiments.yolo_class", lambda: FakeYOLO)
    training = ExperimentConfig(
        action="train",
        data=str(_dataset(tmp_path)), scale=ModelScale.NANO, name="baseline", **_paths(tmp_path)
    )
    storage = ExperimentStorage(training.database_path)
    training_record = storage.create_experiment(training)
    train_model(training, ModelScale.NANO)
    assert storage.metrics(training_record.id, "epoch")[0]["map50_95"] == 0.45
    assert storage.metrics(training_record.id, "summary")[0]["parameters"] == 0

    tuning = ExperimentConfig(
        action="tune",
        data=str(_dataset(tmp_path)), scale=ModelScale.SMALL, name="tune", **_paths(tmp_path)
    )
    tuning_record = storage.create_experiment(tuning)
    tune_model(tuning, ModelScale.SMALL)
    assert storage.metrics(tuning_record.id, "trial")[0]["fitness"] == 0.5
    assert FakeYOLO.calls[-1][1]["use_ray"] is False


def test_completed_training_can_be_evaluated(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("edge_ai.experiments.yolo_class", lambda: FakeYOLO)
    config = ExperimentConfig(
        action="train",
        data=str(_dataset(tmp_path)),
        scale=ModelScale.NANO,
        name="trained",
        **_paths(tmp_path),
    )
    storage = ExperimentStorage(config.database_path)
    training = storage.create_experiment(config)
    train_model(config)
    storage.set_status(training.id, "completed")
    evaluation = create_evaluation(training.id, storage.path)
    evaluate_model(ExperimentConfig.from_dict(evaluation.config))
    assert storage.metrics(evaluation.id, "evaluation")[0]["map50_95"] == 0.55


def test_explicit_promotion_uses_best_hyperparameters(tmp_path: Path) -> None:
    storage = ExperimentStorage(tmp_path / "runs.sqlite3")
    config = ExperimentConfig(
        action="tune",
        data=str(_dataset(tmp_path)), scale=ModelScale.MEDIUM, name="tune", **_paths(tmp_path)
    )
    record = storage.create_experiment(config)
    storage.set_status(record.id, "running", pid=123)
    storage.save_metric(
        record.id, "trial", 1, {"fitness": 0.6, "map50_95": 0.6, "hyperparameters": {"lr0": 0.003}}
    )
    storage.set_status(record.id, "completed")
    promoted = promote_best(record.id, storage.path)
    assert promoted.action == "train"
    assert promoted.parent_id == record.id
    assert promoted.config["extra_args"]["lr0"] == 0.003

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from model_development.config import ExperimentConfig
from model_development.experiments import (
    ONLINE_AUGMENTATION,
    TUNING_SPACE,
    evaluate_model,
    train_model,
    tune_model,
)


class FakeYOLO:
    calls: ClassVar[list[tuple[str, dict]]] = []

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint

    def train(self, **kwargs) -> None:
        self.calls.append(("train", kwargs))

    def tune(self, **kwargs) -> None:
        self.calls.append(("tune", kwargs))

    def val(self, **kwargs):
        self.calls.append(("val", kwargs))
        return SimpleNamespace(results_dict={}, speed={})


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text("train: train\nval: val\nnames: [alfalfa]\n", encoding="utf-8")
    return path


def test_training_and_tuning_use_only_fixed_augmentation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("model_development.experiments.yolo_class", lambda: FakeYOLO)
    FakeYOLO.calls.clear()
    config = ExperimentConfig(
        action="train",
        name="alfalfa",
        data=str(_dataset(tmp_path)),
        hyperparameters={"hsv_h": 0.9, "mosaic": 1.0},
    )

    train_model(config, tmp_path / "train-run")
    tune_model(config, tmp_path / "tune-run")

    train_args = FakeYOLO.calls[0][1]
    tune_args = FakeYOLO.calls[1][1]
    for args in (train_args, tune_args):
        assert {key: args[key] for key in ONLINE_AUGMENTATION} == ONLINE_AUGMENTATION
        assert args["augmentations"] == []
    assert tune_args["use_ray"] is False
    assert tune_args["space"] == TUNING_SPACE
    assert ONLINE_AUGMENTATION.keys().isdisjoint(TUNING_SPACE)


def test_evaluation_does_not_receive_training_augmentation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("model_development.experiments.yolo_class", lambda: FakeYOLO)
    FakeYOLO.calls.clear()
    weights = tmp_path / "best.pt"
    weights.touch()
    config = ExperimentConfig(
        action="evaluate",
        name="evaluate",
        data=str(_dataset(tmp_path)),
        weights=str(weights),
    )

    evaluate_model(config, tmp_path / "evaluate-run")

    args = FakeYOLO.calls[0][1]
    assert ONLINE_AUGMENTATION.keys().isdisjoint(args)
    assert "augmentations" not in args
    assert "space" not in args

from pathlib import Path

import pytest

from model_development.config import (
    ExperimentConfig,
    model_checkpoint,
    validate_device_available,
)


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text("train: train\nval: val\nnames: [target]\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("scale", ["n", "s", "m", "l", "x"])
def test_model_checkpoint(scale: str) -> None:
    assert model_checkpoint(scale) == f"yolov8{scale}.pt"


def test_invalid_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model scale"):
        model_checkpoint("xxl")


def test_config_round_trip(tmp_path: Path) -> None:
    config = ExperimentConfig(action="tune", name="test", data=str(_dataset(tmp_path)), scale="s")
    restored = ExperimentConfig.from_dict(config.to_dict())
    assert restored == config


def test_missing_dataset_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Dataset does not exist"):
        ExperimentConfig(action="train", name="test", data=str(tmp_path / "missing.yaml"))


def test_device_values_and_availability(tmp_path: Path, monkeypatch) -> None:
    dataset = str(_dataset(tmp_path))
    cpu = ExperimentConfig(action="train", name="Jesse's laptop", data=dataset)
    assert cpu.yolo_args()["device"] == "cpu"

    with pytest.raises(ValueError, match="Device must be"):
        ExperimentConfig(action="train", name="test", data=dataset, device="jesselaptop")

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    with pytest.raises(ValueError, match="CUDA device 0 is unavailable"):
        validate_device_available("0")

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    gpu = ExperimentConfig(action="train", name="test", data=dataset, device="0")
    validate_device_available(gpu.device)
    assert gpu.yolo_args()["device"] == "0"


def test_evaluation_requires_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Checkpoint does not exist"):
        ExperimentConfig(action="evaluate", name="test", data=str(_dataset(tmp_path)))

"""Configuration for one YOLO experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MODEL_SCALES = ("n", "s", "m", "l", "x")


def model_checkpoint(scale: str) -> str:
    if scale not in MODEL_SCALES:
        raise ValueError(f"Unknown model scale: {scale}")
    return f"yolov8{scale}.pt"


@dataclass(slots=True)
class ExperimentConfig:
    action: str
    name: str
    data: str
    scale: str = "n"
    device: str = "0"
    imgsz: int = 640
    batch: int = 8
    epochs: int = 300
    iterations: int = 10
    split: str = "test"
    weights: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in {"train", "tune", "evaluate"}:
            raise ValueError(f"Unknown experiment action: {self.action}")
        if not self.name.strip():
            raise ValueError("Experiment name cannot be empty")
        model_checkpoint(self.scale)
        if not Path(self.data).expanduser().is_file():
            raise ValueError(f"Dataset does not exist: {self.data}")
        if self.imgsz < 32 or self.batch < 1 or self.epochs < 1 or self.iterations < 1:
            raise ValueError("Image size, batch, epochs, and iterations must be positive")
        if self.split not in {"val", "test"}:
            raise ValueError("Evaluation split must be 'val' or 'test'")
        if self.action == "evaluate" and not Path(self.weights).expanduser().is_file():
            raise ValueError(f"Checkpoint does not exist: {self.weights}")

    @property
    def checkpoint(self) -> str:
        return self.weights or model_checkpoint(self.scale)

    def yolo_args(self) -> dict[str, Any]:
        args = {
            "data": self.data,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
        }
        if self.action != "evaluate":
            args["epochs"] = self.epochs
        args.update(self.hyperparameters)
        return args

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExperimentConfig:
        return cls(**values)

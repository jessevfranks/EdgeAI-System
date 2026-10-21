"""Configuration for the onboard inference runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# is this class worth having?
@dataclass(frozen=True, slots=True)
class OnboardConfig:
    model: str
    device: str
    imgsz: int
    confidence: float
    iou: float
    storage_root: str
    fps: float

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("FPS must be greater than zero")

    @classmethod
    def from_yaml(cls, path: str | Path) -> OnboardConfig:
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Onboard configuration does not exist: {config_path}")

        try:
            values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid onboard configuration: {error}") from error
        if not isinstance(values, dict):
            raise ValueError("Onboard configuration must be a YAML mapping")

        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid onboard configuration: {error}") from error

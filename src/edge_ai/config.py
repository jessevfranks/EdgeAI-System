"""Simple configuration objects shared by the UI and workers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """A configuration cannot be used for an experiment."""


class ModelScale(str, Enum):
    NANO = "n"
    SMALL = "s"
    MEDIUM = "m"
    LARGE = "l"
    EXTRA_LARGE = "x"

    @classmethod
    def parse(cls, value: str | ModelScale) -> ModelScale:
        try:
            return value if isinstance(value, cls) else cls(value.lower().strip())
        except ValueError as exc:
            raise ConfigurationError(f"Unknown model scale: {value}") from exc


MODEL_CHECKPOINTS = {scale: f"yolov8{scale.value}.pt" for scale in ModelScale}

CORE_WIDE_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "lr0": (1e-5, 5e-2),
    "lrf": (1e-3, 1.0),
    "momentum": (0.70, 0.99),
    "weight_decay": (0.0, 0.002),
    "warmup_epochs": (0.0, 10.0),
    "warmup_momentum": (0.0, 0.95),
    "box": (1.0, 20.0),
    "cls": (0.1, 4.0),
    "dfl": (0.4, 12.0),
}


def model_checkpoint(scale: str | ModelScale) -> str:
    return MODEL_CHECKPOINTS[ModelScale.parse(scale)]


@dataclass(slots=True)
class ExperimentConfig:
    """All experiment settings in one serializable, inspectable object."""

    action: str
    name: str
    data: str
    scale: ModelScale
    profile: str = "research_baseline"
    optimizer: str = "SGD"
    device: str = "0"
    imgsz: int = 640
    batch: int = 8
    epochs: int = 300
    patience: int = 50
    iterations: int = 10
    seed: int = 42
    workers: int = 8
    cache: bool = False
    amp: bool = True
    search_profile: str = "core_wide"
    search_space: dict[str, tuple[float, float]] = field(default_factory=dict)
    split: str = "test"
    weights: str = ""
    parent_id: str | None = None
    dataset_fingerprint: str = ""
    experiment_id: str | None = None
    resume: bool = False
    database_path: Path = Path("runs/experiments.sqlite3")
    run_root: Path = Path("runs")
    artifact_root: Path = Path("artifacts")
    extra_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scale = ModelScale.parse(self.scale)
        self.database_path = Path(self.database_path)
        self.run_root = Path(self.run_root)
        self.artifact_root = Path(self.artifact_root)
        if self.action not in {"train", "tune", "evaluate"}:
            raise ConfigurationError(f"Unknown experiment action: {self.action}")
        if not self.name.strip():
            raise ConfigurationError("Experiment name cannot be empty")
        if self.epochs < 1 or self.iterations < 1 or self.imgsz < 32 or self.batch < 1:
            raise ConfigurationError("Epochs, iterations, batch, and image size must be positive")
        if self.split not in {"val", "test"}:
            raise ConfigurationError("Evaluation split must be 'val' or 'test'")
        self.search_space = {
            key: (float(bounds[0]), float(bounds[1])) for key, bounds in self.search_space.items()
        }
        if any(low >= high for low, high in self.search_space.values()):
            raise ConfigurationError("Every search-space minimum must be below its maximum")

    @property
    def checkpoint(self) -> str:
        return self.weights or model_checkpoint(self.scale)

    @property
    def run_dir(self) -> Path:
        if not self.experiment_id:
            raise ConfigurationError("The experiment must be saved before it can run")
        return self.run_root / self.experiment_id

    def search_bounds(self) -> dict[str, tuple[float, float]] | None:
        if self.search_profile == "ultralytics_default":
            return None
        if self.search_profile == "custom" and not self.search_space:
            raise ConfigurationError("A custom search space cannot be empty")
        return self.search_space or dict(CORE_WIDE_SEARCH_SPACE)

    def yolo_args(self) -> dict[str, Any]:
        args = {
            "data": self.data,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
            "workers": self.workers,
        }
        if self.action != "evaluate":
            args.update(
                epochs=self.epochs,
                patience=self.patience,
                optimizer=self.optimizer,
                seed=self.seed,
                cache=self.cache,
                amp=self.amp,
            )
        args.update(self.extra_args)
        return args

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["scale"] = self.scale.value
        for key in ("database_path", "run_root", "artifact_root"):
            result[key] = str(result[key])
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExperimentConfig:
        return cls(**values)


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise ConfigurationError(f"File does not exist: {source}")
    value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"Expected a YAML mapping in {source}")
    return value


def load_dataset_yaml(path: str | Path, *, require_test: bool = False) -> dict[str, Any]:
    dataset = load_yaml(Path(path).expanduser().resolve())
    required = ["train", "val", "names"] + (["test"] if require_test else [])
    missing = [key for key in required if not dataset.get(key)]
    if missing:
        raise ConfigurationError(f"Dataset YAML is missing: {', '.join(missing)}")
    return dataset


def dataset_fingerprint(path: str | Path) -> str:
    """Identify the dataset definition used for an experiment."""
    source = Path(path).expanduser().resolve()
    load_dataset_yaml(source)
    return sha256(source.read_bytes()).hexdigest()


def merged_profile(settings: dict[str, Any], profile_name: str) -> dict[str, Any]:
    try:
        profile = settings["profiles"][profile_name]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown profile: {profile_name}") from exc
    return {**settings.get("defaults", {}), **profile}

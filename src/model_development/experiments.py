"""Direct Ultralytics train, tune, and evaluate operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .metrics import result_metrics


def yolo_class() -> Any:
    try:
        from ultralytics import YOLO

        return YOLO
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies before running an experiment") from exc


def train_model(config: ExperimentConfig, run_dir: Path) -> None:
    model = yolo_class()(config.checkpoint)
    model.train(
        **config.yolo_args(),
        project=str(run_dir),
        name="train",
        exist_ok=True,
        save=True,
        plots=True,
    )


def tune_model(config: ExperimentConfig, run_dir: Path) -> None:
    """Run Ultralytics' built-in genetic hyperparameter tuner."""
    model = yolo_class()(config.checkpoint)
    model.tune(
        **config.yolo_args(),
        iterations=config.iterations,
        use_ray=False,
        project=str(run_dir),
        name="tune",
        exist_ok=True,
        save=True,
        plots=True,
        val=True,
    )


def evaluate_model(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    model = yolo_class()(config.checkpoint)
    result = model.val(
        **config.yolo_args(),
        split=config.split,
        project=str(run_dir),
        name="evaluate",
        exist_ok=True,
        plots=True,
    )
    return result_metrics(result)

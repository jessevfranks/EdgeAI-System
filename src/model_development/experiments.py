"""Direct Ultralytics train, tune, and evaluate operations."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from .config import ExperimentConfig
from .metrics import result_metrics

ONLINE_AUGMENTATION = {
    "hsv_h": 0.022, # ~8%
    "hsv_s": 0.25,
    "hsv_v": 0.25,
    "scale": 0.20,
    "shear": 7.0,
    "degrees": 0.0,
    "translate": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
}

TUNING_SPACE = {
    "lr0": (1e-5, 1e-2),
    "lrf": (0.01, 1.0),
    "momentum": (0.7, 0.98, 0.3),
    "weight_decay": (0.0, 0.001),
    "warmup_epochs": (0.0, 5.0),
    "warmup_momentum": (0.0, 0.95),
    "box": (1.0, 20.0),
    "cls": (0.1, 4.0),
    "cls_pw": (0.0, 1.0),
    "dfl": (0.4, 12.0),
}


def training_args(config: ExperimentConfig) -> dict[str, Any]:
    """Return training settings with the fixed augmentation policy applied last."""
    args = config.yolo_args()
    args.update(ONLINE_AUGMENTATION)
    args["augmentations"] = []
    return args


def yolo_class():
    """Return the Ultralytics model class, allowing tests to replace it."""
    return YOLO


def move_checkpoints(run_dir: Path, artifact_dir: Path) -> None:
    """Keep Ultralytics' checkpoint paths relative to the experiment run."""
    for checkpoint in run_dir.rglob("*.pt"):
        target = artifact_dir / checkpoint.relative_to(run_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(checkpoint, target)


def train_model(config: ExperimentConfig, run_dir: Path) -> None:
    model = yolo_class()(config.checkpoint)
    model.train(
        **training_args(config),
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
        **training_args(config),
        space=TUNING_SPACE,
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

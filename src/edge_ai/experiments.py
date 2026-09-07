"""The three Ultralytics operations launched by the Streamlit UI."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, ModelScale, load_dataset_yaml
from .metrics import environment_info, newest, result_metrics, sync_metrics
from .storage import ExperimentRecord, ExperimentStorage


def yolo_class() -> Any:
    try:
        from ultralytics import YOLO

        return YOLO
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies before running an experiment") from exc


def prepare(config: ExperimentConfig) -> tuple[ExperimentStorage, ExperimentRecord]:
    load_dataset_yaml(config.data, require_test=config.action == "evaluate" and config.split == "test")
    storage = ExperimentStorage(config.database_path)
    if not config.experiment_id:
        config.experiment_id = storage.create_experiment(config).id
    config.run_dir.mkdir(parents=True, exist_ok=True)
    return storage, storage.get(config.experiment_id)


def model_size(model: Any) -> int | None:
    try:
        return sum(parameter.numel() for parameter in model.model.parameters())
    except AttributeError:
        return None


def write_manifest(config: ExperimentConfig, parameters: int | None) -> None:
    manifest = {
        "config": config.to_dict(),
        "environment": environment_info(),
        "model_parameters": parameters,
    }
    (config.run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def save_summary(
    storage: ExperimentStorage,
    record: ExperimentRecord,
    parameters: int | None,
    started: float,
) -> None:
    summary: dict[str, Any] = {
        "duration_seconds": time.perf_counter() - started,
        "parameters": parameters,
    }
    if record.best_checkpoint:
        summary["checkpoint_size_bytes"] = Path(record.best_checkpoint).stat().st_size
    try:
        import torch

        if torch.cuda.is_available():
            summary["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        pass
    storage.save_metric(record.id, "summary", 0, summary)


def train_model(
    config: ExperimentConfig, scale: ModelScale | None = None, *, resume: bool | None = None
) -> ExperimentRecord:
    """Fine-tune one YOLOv8 model scale."""
    config.scale = scale or config.scale
    storage, record = prepare(config)
    started = time.perf_counter()
    previous = newest(config.run_dir, "last.pt") if (config.resume if resume is None else resume) else None
    model = yolo_class()(str(previous) if previous else config.checkpoint)
    parameters = model_size(model)
    write_manifest(config, parameters)
    if previous:
        model.train(resume=True)
    else:
        model.train(
            **config.yolo_args(),
            project=str(config.run_dir),
            name="train",
            exist_ok=True,
            save=True,
            plots=True,
        )
    record = sync_metrics(storage, record)
    save_summary(storage, record, parameters, started)
    if config.profile == "final_train" and record.best_checkpoint:
        destination = config.artifact_root / "models" / config.scale.value / record.id
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.best_checkpoint, destination / "best.pt")
        shutil.copy2(config.run_dir / "manifest.json", destination / "manifest.json")
    return storage.get(record.id)


def tune_model(
    config: ExperimentConfig, scale: ModelScale | None = None, *, resume: bool | None = None
) -> ExperimentRecord:
    """Run Ultralytics' built-in genetic hyperparameter tuner."""
    config.scale = scale or config.scale
    storage, record = prepare(config)
    started = time.perf_counter()
    model = yolo_class()(config.checkpoint)
    parameters = model_size(model)
    write_manifest(config, parameters)
    arguments = {
        **config.yolo_args(),
        "iterations": config.iterations,
        "use_ray": False,
        "project": str(config.run_dir),
        "name": "tune",
        "exist_ok": True,
        "resume": config.resume if resume is None else resume,
        "cleanup": True,
        "save": True,
        "plots": True,
        "val": True,
    }
    if (bounds := config.search_bounds()) is not None:
        arguments["space"] = bounds
    model.tune(**arguments)
    record = sync_metrics(storage, record)
    save_summary(storage, record, parameters, started)
    return record


def evaluate_model(config: ExperimentConfig, experiment_id: str | None = None) -> ExperimentRecord:
    """Evaluate a selected checkpoint on an explicitly chosen split."""
    config.experiment_id = experiment_id or config.experiment_id
    storage, record = prepare(config)
    started = time.perf_counter()
    weights = Path(config.weights).expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
    model = yolo_class()(str(weights))
    parameters = model_size(model)
    write_manifest(config, parameters)
    result = model.val(
        **config.yolo_args(),
        split=config.split,
        project=str(config.run_dir),
        name="evaluate",
        exist_ok=True,
        plots=True,
    )
    metrics = result_metrics(result)
    storage.save_metric(record.id, "evaluation", 0, metrics)
    storage.update_progress(record.id, fitness=metrics.get("fitness"))
    save_summary(storage, record, parameters, started)
    return storage.get(record.id)


def promote_best(
    tuning_experiment_id: str, database_path: str | Path = "runs/experiments.sqlite3"
) -> ExperimentRecord:
    """Create a full training job from a completed tuning campaign."""
    storage = ExperimentStorage(database_path)
    source = storage.get(tuning_experiment_id)
    trials = storage.metrics(source.id, "trial")
    if source.action != "tune" or source.status != "completed" or not trials:
        raise ValueError("Select a completed tuning experiment with at least one trial")
    scored_trials = [trial for trial in trials if isinstance(trial.get("fitness"), (int, float))]
    if not scored_trials:
        raise ValueError("The tuning experiment does not have a scored trial")
    best = max(scored_trials, key=lambda trial: trial["fitness"])
    original = ExperimentConfig.from_dict(source.config)
    promoted = ExperimentConfig(
        action="train",
        name=f"{source.name} final",
        data=original.data,
        scale=original.scale,
        profile="final_train",
        optimizer=original.optimizer,
        device=original.device,
        imgsz=original.imgsz,
        batch=original.batch,
        epochs=300,
        patience=50,
        seed=original.seed,
        workers=original.workers,
        cache=original.cache,
        amp=original.amp,
        parent_id=source.id,
        database_path=Path(database_path),
        run_root=original.run_root,
        artifact_root=original.artifact_root,
        extra_args=best.get("hyperparameters", {}),
    )
    return storage.create_experiment(promoted)


def create_evaluation(
    source_experiment_id: str,
    database_path: str | Path = "runs/experiments.sqlite3",
    *,
    split: str = "test",
) -> ExperimentRecord:
    storage = ExperimentStorage(database_path)
    source = storage.get(source_experiment_id)
    if source.status != "completed" or not source.best_checkpoint:
        raise ValueError("Select completed training with a best checkpoint")
    original = ExperimentConfig.from_dict(source.config)
    return storage.create_experiment(
        ExperimentConfig(
            action="evaluate",
            name=f"{source.name} {split}",
            data=original.data,
            scale=original.scale,
            device=original.device,
            imgsz=original.imgsz,
            batch=original.batch,
            split=split,
            weights=source.best_checkpoint,
            parent_id=source.id,
            database_path=Path(database_path),
            run_root=original.run_root,
            artifact_root=original.artifact_root,
        )
    )

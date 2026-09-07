"""Read Ultralytics output files into the small experiment database."""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from .storage import ExperimentRecord, ExperimentStorage

ALIASES = {
    "metrics/precision(b)": "precision",
    "metrics/recall(b)": "recall",
    "metrics/map50(b)": "map50",
    "metrics/map50-95(b)": "map50_95",
    "train/box_loss": "train_box_loss",
    "train/cls_loss": "train_cls_loss",
    "train/dfl_loss": "train_dfl_loss",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "val/dfl_loss": "val_dfl_loss",
    "time": "duration_seconds",
}


def number(value: Any) -> Any:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(result):
        return None
    return int(result) if result.is_integer() else result


def normalize_metrics(values: dict[str, Any]) -> dict[str, Any]:
    result = {
        ALIASES.get(key.strip().lower(), key.strip().replace("/", "_")): number(value)
        for key, value in values.items()
        if value not in (None, "")
    }
    precision, recall = result.get("precision"), result.get("recall")
    if isinstance(precision, (int, float)) and isinstance(recall, (int, float)) and precision + recall:
        result["f1"] = 2 * precision * recall / (precision + recall)
    if "fitness" not in result and "map50" in result and "map50_95" in result:
        result["fitness"] = 0.1 * result["map50"] + 0.9 * result["map50_95"]
    return result


def read_results_csv(path: str | Path) -> list[dict[str, Any]]:
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            return [normalize_metrics({key.strip(): value for key, value in row.items()}) for row in csv.DictReader(stream)]
    except (OSError, csv.Error):
        return []


def read_tune_ndjson(path: str | Path) -> list[dict[str, Any]]:
    """Ignore an unfinished final line left when a tuning worker is stopped."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    trials = []
    for line in lines:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        datasets = raw.get("datasets", {})
        metrics = normalize_metrics(next(iter(datasets.values()), {}) or {})
        metrics["fitness"] = number(raw.get("fitness"))
        metrics["hyperparameters"] = raw.get("hyperparameters", {})
        trials.append({"iteration": int(raw.get("iteration", len(trials) + 1)), **metrics})
    return trials


def newest(root: Path, filename: str) -> Path | None:
    matches = list(root.rglob(filename))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def sync_metrics(storage: ExperimentStorage, record: ExperimentRecord) -> ExperimentRecord:
    """Idempotently copy completed rows from Ultralytics files into SQLite."""
    root = Path(record.run_dir)
    if record.action == "train":
        source = newest(root, "results.csv")
        if source:
            rows = read_results_csv(source)
            for index, row in enumerate(rows, start=1):
                step = int(row.pop("epoch", index))
                storage.save_metric(record.id, "epoch", step, row)
            fitness = [row["fitness"] for row in rows if isinstance(row.get("fitness"), (int, float))]
            if fitness:
                storage.update_progress(record.id, fitness=max(fitness))

    if record.action == "tune":
        source = newest(root, "tune_results.ndjson")
        if source:
            trials = read_tune_ndjson(source)
            for trial in trials:
                step = trial.pop("iteration")
                storage.save_metric(record.id, "trial", step, trial)
            fitness = [trial["fitness"] for trial in trials if isinstance(trial.get("fitness"), (int, float))]
            storage.update_progress(
                record.id,
                step=len(trials),
                fitness=max(fitness) if fitness else None,
            )
    return storage.get(record.id)


def result_metrics(result: Any) -> dict[str, Any]:
    values = normalize_metrics(dict(getattr(result, "results_dict", {}) or {}))
    values.update({f"speed_{key}_ms": number(value) for key, value in (getattr(result, "speed", {}) or {}).items()})
    return values


def environment_info() -> dict[str, Any]:
    """Small reproducibility snapshot written into each run manifest."""
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("ultralytics", "torch"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    try:
        result["git_revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        ).stdout.strip()
    except OSError:
        result["git_revision"] = ""
    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    except (ImportError, RuntimeError):
        result["cuda_available"] = False
    return result

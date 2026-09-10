"""Background process for one experiment."""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .experiments import evaluate_model, train_model, tune_model
from .metrics import sync_metrics
from .storage import ExperimentStorage

Workflow = Callable[[ExperimentConfig, Path], dict[str, Any] | None]
WORKFLOWS: dict[str, Workflow] = {
    "train": train_model,
    "tune": tune_model,
    "evaluate": evaluate_model,
}


def run_worker(database_path: str | Path, experiment_id: str) -> int:
    storage = ExperimentStorage(database_path)
    record = storage.get(experiment_id)
    try:
        config = ExperimentConfig.from_dict(record.config)
        storage.set_status(experiment_id, "running")
        result = WORKFLOWS[config.action](config, Path(record.run_dir))
        if result:
            storage.save_metric(experiment_id, "evaluation", 0, result)
        sync_metrics(storage, record)
        storage.set_status(experiment_id, "completed")
        return 0
    except Exception as exc:  # noqa: BLE001 - record failures from the worker boundary
        traceback.print_exc()
        storage.set_status(experiment_id, "failed", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(run_worker(sys.argv[1], sys.argv[2]))

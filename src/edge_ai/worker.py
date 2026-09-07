"""Internal process used so training survives Streamlit page refreshes."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from .config import ExperimentConfig
from .experiments import evaluate_model, train_model, tune_model
from .jobs import JobManager
from .storage import ExperimentStorage


def run_worker(database_path: str | Path, experiment_id: str) -> int:
    storage = ExperimentStorage(database_path)
    config = ExperimentConfig.from_dict(storage.get(experiment_id).config)
    try:
        {"train": train_model, "tune": tune_model, "evaluate": evaluate_model}[config.action](config)
        storage.set_status(experiment_id, "completed")
        return 0
    except Exception as exc:  # noqa: BLE001 - process boundary records the useful message
        traceback.print_exc()
        storage.set_status(experiment_id, "failed", error=str(exc))
        return 1
    finally:
        JobManager(database_path).start_waiting_jobs()


if __name__ == "__main__":
    raise SystemExit(run_worker(sys.argv[1], sys.argv[2]))

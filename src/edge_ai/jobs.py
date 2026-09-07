"""Launch, queue, stop, and resume the UI's background workers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psutil

from .config import ExperimentConfig
from .metrics import sync_metrics
from .storage import ExperimentRecord, ExperimentStorage


class JobManager:
    def __init__(self, database_path: str | Path = "runs/experiments.sqlite3") -> None:
        self.storage = ExperimentStorage(database_path)

    def enqueue(self, experiment_id: str) -> ExperimentRecord:
        self.start_waiting_jobs()
        return self.storage.get(experiment_id)

    def start_waiting_jobs(self) -> None:
        """Start one queued job per free device."""
        record = self.storage.claim_next()
        while record:
            try:
                self._spawn(record)
            except OSError as exc:
                self.storage.set_status(record.id, "failed", error=str(exc))
            record = self.storage.claim_next()

    def _spawn(self, record: ExperimentRecord) -> None:
        log_path = Path(record.run_dir) / "worker.log"
        command = [
            sys.executable,
            "-u",
            "-m",
            "edge_ai.worker",
            str(self.storage.path),
            record.id,
        ]
        options = {"stdin": subprocess.DEVNULL, "cwd": Path(record.run_dir).parents[1]}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            options["start_new_session"] = True
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **options)
        self.storage.set_status(record.id, "running", pid=process.pid)

    def stop(self, experiment_id: str) -> ExperimentRecord:
        record = self.storage.get(experiment_id)
        if record.status == "queued":
            return self.storage.set_status(record.id, "stopped")
        if record.status != "running":
            return record
        if record.pid and psutil.pid_exists(record.pid):
            try:
                parent = psutil.Process(record.pid)
                processes = parent.children(recursive=True) + [parent]
                for process in processes:
                    process.terminate()
                _, alive = psutil.wait_procs(processes, timeout=5)
                for process in alive:
                    process.kill()
            except psutil.Error:
                pass
        sync_metrics(self.storage, record)
        stopped = self.storage.set_status(record.id, "stopped")
        self.start_waiting_jobs()
        return stopped

    def resume(self, experiment_id: str) -> ExperimentRecord:
        record = self.storage.get(experiment_id)
        if record.status not in {"stopped", "interrupted", "failed"}:
            raise ValueError(f"Cannot resume a {record.status} experiment")
        config = ExperimentConfig.from_dict(record.config)
        config.resume = True
        self.storage.save_config(config)
        self.storage.set_status(record.id, "queued")
        return self.enqueue(record.id)

    def reconcile(self) -> None:
        """Recover jobs whose worker disappeared while the UI was closed."""
        for record in self.storage.list(statuses=["running"]):
            sync_metrics(self.storage, record)
            if not record.pid or not psutil.pid_exists(record.pid):
                self.storage.set_status(record.id, "interrupted")
        self.start_waiting_jobs()

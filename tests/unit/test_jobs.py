import os
from pathlib import Path
from types import SimpleNamespace

from edge_ai.config import ExperimentConfig, ModelScale
from edge_ai.jobs import JobManager


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text("train: train\nval: val\ntest: test\nnames: [target]\n", encoding="utf-8")
    return path


def _config(tmp_path: Path, name: str, device: str) -> ExperimentConfig:
    return ExperimentConfig(
        action="train",
        data=str(_dataset(tmp_path)),
        scale=ModelScale.NANO,
        name=name,
        device=device,
        database_path=tmp_path / "experiments.sqlite3",
        run_root=tmp_path / "runs",
        artifact_root=tmp_path / "artifacts",
    )


def test_scheduler_prevents_equivalent_device_strings_from_overlapping(tmp_path: Path, monkeypatch) -> None:
    manager = JobManager(tmp_path / "experiments.sqlite3")
    first = manager.storage.create_experiment(_config(tmp_path, "first", "0"))
    second = manager.storage.create_experiment(_config(tmp_path, "second", "cuda:0"))

    monkeypatch.setattr("edge_ai.jobs.subprocess.Popen", lambda *args, **kwargs: SimpleNamespace(pid=os.getpid()))
    manager.start_waiting_jobs()

    assert manager.storage.get(first.id).status == "running"
    assert manager.storage.get(second.id).status == "queued"


def test_resume_preserves_config_and_marks_resume_protocol(tmp_path: Path, monkeypatch) -> None:
    manager = JobManager(tmp_path / "experiments.sqlite3")
    record = manager.storage.create_experiment(_config(tmp_path, "resume", "1"))
    manager.storage.set_status(record.id, "stopped")
    monkeypatch.setattr("edge_ai.jobs.subprocess.Popen", lambda *args, **kwargs: SimpleNamespace(pid=os.getpid()))

    resumed = manager.resume(record.id)

    assert resumed.status == "running"
    assert manager.storage.get(record.id).config["resume"] is True


def test_stop_terminates_the_worker_and_marks_job_stopped(tmp_path: Path, monkeypatch) -> None:
    manager = JobManager(tmp_path / "experiments.sqlite3")
    record = manager.storage.create_experiment(_config(tmp_path, "stop", "0"))
    manager.storage.set_status(record.id, "running", pid=123)
    process = SimpleNamespace(children=lambda recursive: [], terminate=lambda: None, kill=lambda: None)
    monkeypatch.setattr("edge_ai.jobs.psutil.pid_exists", lambda pid: True)
    monkeypatch.setattr("edge_ai.jobs.psutil.Process", lambda pid: process)
    monkeypatch.setattr("edge_ai.jobs.psutil.wait_procs", lambda processes, timeout: (processes, []))

    stopped = manager.stop(record.id)

    assert stopped.status == "stopped"

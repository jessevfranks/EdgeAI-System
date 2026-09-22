import os
from pathlib import Path
from types import SimpleNamespace

import torch
from streamlit.testing.v1 import AppTest

from model_development.config import ExperimentConfig
from model_development.dashboard import (
    _evaluation_config,
    _promoted_config,
    _start_worker,
)
from model_development.storage import ExperimentStorage


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text("train: train\nval: val\ntest: test\nnames: [target]\n", encoding="utf-8")
    return path


def test_dashboard_pages_render() -> None:
    dashboard = Path(__file__).resolve().parents[2] / "src" / "model_development" / "dashboard.py"
    app = AppTest.from_file(str(dashboard)).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "EdgeAI YOLOv8 Experiments"
    app.sidebar.radio[0].set_value("Experiments")
    app.run(timeout=20)
    assert not app.exception


def test_device_options_follow_available_gpus(monkeypatch) -> None:
    dashboard = Path(__file__).resolve().parents[2] / "src" / "model_development" / "dashboard.py"
    for gpu_count, expected in ((0, ["CPU"]), (1, ["CPU", "GPU 0"])):
        monkeypatch.setattr(torch.cuda, "is_available", lambda count=gpu_count: count > 0)
        monkeypatch.setattr(torch.cuda, "device_count", lambda count=gpu_count: count)
        app = AppTest.from_file(str(dashboard)).run(timeout=20)
        assert not app.exception
        device = next(widget for widget in app.selectbox if widget.label == "Device")
        assert device.options == expected
        assert device.value == "cpu"


def test_completed_run_configs_are_simple(tmp_path: Path) -> None:
    storage = ExperimentStorage(tmp_path / "experiments.sqlite3")
    tuning = storage.create_experiment(
        ExperimentConfig(action="tune", name="tune", data=str(_dataset(tmp_path)), epochs=20)
    )
    promoted = _promoted_config(tuning, {"hyperparameters": {"lr0": 0.005}})
    assert promoted.action == "train"
    assert promoted.epochs == 300
    assert promoted.hyperparameters == {"lr0": 0.005}

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"best")
    evaluation = _evaluation_config(tuning, str(checkpoint), "test")
    assert evaluation.action == "evaluate"
    assert evaluation.weights == str(checkpoint)


def test_start_worker_creates_log(tmp_path: Path, monkeypatch) -> None:
    storage = ExperimentStorage(tmp_path / "experiments.sqlite3")
    record = storage.create_experiment(
        ExperimentConfig(action="train", name="train", data=str(_dataset(tmp_path)))
    )
    launched = {}

    def fake_popen(*args, **kwargs):
        launched.update(kwargs)
        return SimpleNamespace(pid=123)

    monkeypatch.setattr("model_development.dashboard.subprocess.Popen", fake_popen)
    _start_worker(storage, record)
    assert (Path(record.run_dir) / "worker.log").exists()
    assert launched["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(
        Path(__file__).resolve().parents[2] / "src"
    )

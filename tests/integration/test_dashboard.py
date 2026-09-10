from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from model_development.config import ExperimentConfig
from model_development.dashboard import _evaluation_config, _promoted_config, _start_worker
from model_development.storage import ExperimentStorage


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text("train: train\nval: val\ntest: test\nnames: [target]\n", encoding="utf-8")
    return path


def test_dashboard_pages_render() -> None:
    dashboard = Path(__file__).resolve().parents[2] / "src" / "edge_ai" / "dashboard.py"
    app = AppTest.from_file(str(dashboard)).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "EdgeAI YOLOv8 Experiments"
    app.sidebar.radio[0].set_value("Experiments")
    app.run(timeout=20)
    assert not app.exception


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
    monkeypatch.setattr(
        "edge_ai.dashboard.subprocess.Popen", lambda *args, **kwargs: SimpleNamespace(pid=123)
    )
    _start_worker(storage, record)
    assert (Path(record.run_dir) / "worker.log").exists()

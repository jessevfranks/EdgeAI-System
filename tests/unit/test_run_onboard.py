from pathlib import Path
from types import SimpleNamespace

import onboard
from scripts import run_onboard


def test_test_images_command_runs_directory_source(monkeypatch, tmp_path: Path, capsys) -> None:
    events: list[object] = []
    metrics_instances: list[object] = []
    config = SimpleNamespace(
        model="model.pt",
        storage_root="data/onboard",
        device="cpu",
        imgsz=320,
        confidence=0.4,
        iou=0.6,
        fps=3,
    )

    class FakeConfig:
        @classmethod
        def from_yaml(cls, path):
            events.append(("config", path))
            return config

    class FakeSource:
        def __init__(self, directory, fps):
            events.append(("source", directory, fps))

    class FakeOrchestrator:
        def __init__(self, **settings):
            events.append(("orchestrator", settings))

        def run_stream(self, source):
            events.append(("run_stream", source))
            yield SimpleNamespace(
                acquired_frame=SimpleNamespace(metadata=SimpleNamespace(frame_id=7)),
                inference_result=SimpleNamespace(detections=(object(), object())),
            )

    class FakeMetrics:
        def __init__(self, storage_root, device):
            events.append(("metrics", storage_root, device))
            metrics_instances.append(self)

    monkeypatch.setattr(onboard, "OnboardConfig", FakeConfig)
    monkeypatch.setattr(onboard, "PacedImageDirectory", FakeSource)
    monkeypatch.setattr(onboard, "MetricsCollector", FakeMetrics)
    monkeypatch.setattr(onboard, "OnboardOrchestrator", FakeOrchestrator)
    config_path = tmp_path / "settings.yaml"

    exit_code = run_onboard.main(
        ["test-images", str(tmp_path), "--config", str(config_path)]
    )

    assert exit_code == 0
    assert events[:4] == [
        ("config", config_path),
        ("source", tmp_path, 3),
        ("metrics", "data/onboard", "cpu"),
        (
            "orchestrator",
            {
                "model_path": "model.pt",
                "storage_root": "data/onboard",
                "device": "cpu",
                "imgsz": 320,
                "confidence": 0.4,
                "iou": 0.6,
                "metrics": metrics_instances[0],
            },
        ),
    ]
    assert capsys.readouterr().out == "Processed frame 7: 2 detections\n"


def test_test_images_command_treats_keyboard_interrupt_as_clean_stop(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    config = SimpleNamespace(
        model="model.pt",
        storage_root="data/onboard",
        device="cpu",
        imgsz=320,
        confidence=0.4,
        iou=0.6,
        fps=3,
    )

    monkeypatch.setattr(
        onboard.OnboardConfig,
        "from_yaml",
        classmethod(lambda cls, path: config),
    )
    monkeypatch.setattr(onboard, "PacedImageDirectory", lambda directory, fps: object())

    class InterruptedOrchestrator:
        def __init__(self, **settings):
            pass

        def run_stream(self, source):
            raise KeyboardInterrupt

    monkeypatch.setattr(onboard, "OnboardOrchestrator", InterruptedOrchestrator)

    exit_code = run_onboard.main(["test-images", str(tmp_path)])

    assert exit_code == 0
    assert capsys.readouterr().out == "Stopped.\n"

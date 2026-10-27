from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import onboard.orchestrator as orchestrator_module
from onboard import OnboardOrchestrator, OrchestrationResult, StreamFrame


def install_fake_stages(monkeypatch, events: list[object], outputs: tuple[object, ...]):
    acquired, processed, inferred, vari = outputs

    class FakeAcquirer:
        def __init__(self, storage_root):
            events.append(("construct_acquirer", storage_root))

        def acquire(self, image, captured_at):
            events.append(("acquire", image, captured_at))
            return acquired

    class FakeProcessor:
        def __init__(self, processors=(), storage_root="data/onboard"):
            events.append(("construct_processor", tuple(processors), storage_root))

        def process(self, frame):
            events.append(("process", frame))
            return processed

    class FakeInference:
        def __init__(self, **settings):
            events.append(("construct_inference", settings))

        def infer(self, frame):
            events.append(("infer", frame))
            return inferred

    class FakeVari:
        def __init__(self, storage_root):
            events.append(("construct_vari", storage_root))

        def compute(self, result):
            events.append(("compute", result))
            return vari

    monkeypatch.setattr(orchestrator_module, "ImageAcquirer", FakeAcquirer)
    monkeypatch.setattr(orchestrator_module, "ImagePipeline", FakeProcessor)
    monkeypatch.setattr(orchestrator_module, "InferencePipeline", FakeInference)
    monkeypatch.setattr(orchestrator_module, "VariCalculator", FakeVari)


def test_orchestrator_constructs_and_runs_stages_in_order(monkeypatch, tmp_path: Path):
    events: list[object] = []
    outputs = (object(), object(), object(), object())
    install_fake_stages(monkeypatch, events, outputs)
    model_path = tmp_path / "model.engine"
    image = Image.new("RGB", (1, 1))
    captured_at = datetime(2026, 10, 15, tzinfo=UTC)

    orchestrator = OnboardOrchestrator(
        model_path,
        storage_root=tmp_path,
        device="1",
        imgsz=320,
        confidence=0.4,
        iou=0.6,
    )
    result = orchestrator.run(image, captured_at)

    assert events == [
        ("construct_acquirer", tmp_path),
        ("construct_processor", (), tmp_path),
        (
            "construct_inference",
            {
                "model_path": model_path,
                "device": "1",
                "imgsz": 320,
                "confidence": 0.4,
                "iou": 0.6,
                "storage_root": tmp_path,
                "metrics": None,
            },
        ),
        ("construct_vari", tmp_path),
        ("acquire", image, captured_at),
        ("process", outputs[0]),
        ("infer", outputs[1]),
        ("compute", outputs[2]),
    ]
    assert result == OrchestrationResult(*outputs)


def test_orchestrator_stops_when_a_stage_fails(monkeypatch, tmp_path: Path):
    events: list[object] = []
    outputs = (object(), object(), object(), object())
    install_fake_stages(monkeypatch, events, outputs)
    orchestrator = OnboardOrchestrator(tmp_path / "model.engine", tmp_path)

    def fail(frame):
        events.append(("infer_failed", frame))
        raise RuntimeError("inference failed")

    orchestrator.inference.infer = fail
    image = Image.new("RGB", (1, 1))

    with pytest.raises(RuntimeError, match="inference failed"):
        orchestrator.run(image)

    assert events[-3:] == [
        ("acquire", image, None),
        ("process", outputs[0]),
        ("infer_failed", outputs[1]),
    ]
    assert not any(event[0] == "compute" for event in events)


def test_orchestrator_runs_stream_frames_in_order(monkeypatch, tmp_path: Path):
    events: list[object] = []
    outputs = (object(), object(), object(), object())
    install_fake_stages(monkeypatch, events, outputs)
    orchestrator = OnboardOrchestrator(tmp_path / "model.engine", tmp_path)
    first = Image.new("RGB", (1, 1), "red")
    second = Image.new("RGB", (1, 1), "blue")
    first_time = datetime(2026, 10, 20, 12, tzinfo=UTC)
    second_time = datetime(2026, 10, 20, 13, tzinfo=UTC)
    frames = (
        StreamFrame(first, first_time, tmp_path / "first.png"),
        StreamFrame(second, second_time, tmp_path / "second.png"),
    )

    results = list(orchestrator.run_stream(frames))

    assert results == [OrchestrationResult(*outputs), OrchestrationResult(*outputs)]
    acquire_events = [event for event in events if event[0] == "acquire"]
    assert acquire_events == [
        ("acquire", first, first_time),
        ("acquire", second, second_time),
    ]


def test_stream_uses_one_metrics_run_for_all_frames(monkeypatch, tmp_path: Path):
    events: list[object] = []
    acquired = SimpleNamespace(metadata=SimpleNamespace(frame_id=4))
    outputs = (acquired, object(), object(), object())
    install_fake_stages(monkeypatch, events, outputs)

    class FakeMetrics:
        @contextmanager
        def run(self):
            events.append("metrics_start")
            try:
                yield
            finally:
                events.append("metrics_finish")

        def record_frame(self, **values):
            events.append(("metrics_frame", values))

    metrics = FakeMetrics()
    orchestrator = OnboardOrchestrator(
        tmp_path / "model.engine",
        tmp_path,
        metrics=metrics,  # type: ignore[arg-type]
    )
    frames = (
        StreamFrame(Image.new("RGB", (1, 1)), datetime.now(UTC), tmp_path / "1.png"),
        StreamFrame(Image.new("RGB", (1, 1)), datetime.now(UTC), tmp_path / "2.png"),
    )

    results = list(orchestrator.run_stream(frames))

    assert results == [OrchestrationResult(*outputs), OrchestrationResult(*outputs)]
    assert events.count("metrics_start") == 1
    assert events.count("metrics_finish") == 1
    metric_frames = [
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "metrics_frame"
    ]
    assert len(metric_frames) == 2
    assert all(event[1]["frame_id"] == 4 for event in metric_frames)

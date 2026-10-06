import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from onboard import Detection, ImageAcquirer, InferencePipeline


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, coordinates, confidences, class_ids) -> None:
        self.xyxy = FakeTensor(coordinates)
        self.conf = FakeTensor(confidences)
        self.cls = FakeTensor(class_ids)

    def __len__(self) -> int:
        return len(self.conf.values)


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None) -> None:
        self.boxes = boxes
        self.names = {0: "weed", 1: "crop"}

    def plot(self):
        plotted = np.zeros((2, 3, 3), dtype=np.uint8)
        plotted[0, 0] = [10, 20, 30]
        return plotted


class FakeModel:
    def __init__(self, result: FakeResult) -> None:
        self.result = result
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def acquired_frame(tmp_path: Path):
    return ImageAcquirer(tmp_path / "onboard").acquire(
        Image.new("RGB", (3, 2), (1, 2, 3)),
        datetime(2026, 10, 6, 12, tzinfo=UTC),
    )


def install_fake_model(monkeypatch, result: FakeResult):
    constructed = []
    model = FakeModel(result)

    def fake_yolo(engine_path: str, task: str):
        constructed.append((engine_path, task))
        return model

    monkeypatch.setattr("onboard.inference.yolo_class", lambda: fake_yolo)
    return model, constructed


def test_engine_path_must_be_an_existing_engine_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.engine"):
        InferencePipeline(tmp_path / "model.pt")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        InferencePipeline(tmp_path / "missing.engine")


def test_infer_returns_and_stores_all_detections(monkeypatch, tmp_path: Path) -> None:
    frame = acquired_frame(tmp_path)
    engine_path = tmp_path / "models" / "detector.engine"
    engine_path.parent.mkdir()
    engine_path.touch()
    boxes = FakeBoxes(
        [[1.0, 2.0, 10.0, 20.0], [3.5, 4.5, 11.5, 21.5]],
        [0.9, 0.75],
        [1.0, 0.0],
    )
    model, constructed = install_fake_model(monkeypatch, FakeResult(boxes))

    pipeline = InferencePipeline(
        engine_path,
        device="1",
        imgsz=320,
        confidence=0.4,
        iou=0.6,
        storage_root=tmp_path / "onboard",
    )
    result = pipeline.infer(frame)

    assert constructed == [(str(engine_path.resolve()), "detect")]
    assert model.calls == [
        {
            "source": str(frame.processed_path),
            "device": "1",
            "imgsz": 320,
            "conf": 0.4,
            "iou": 0.6,
            "verbose": False,
            "save": False,
        }
    ]
    assert result.detections == (
        Detection(1.0, 2.0, 10.0, 20.0, 1, "crop", 0.9),
        Detection(3.5, 4.5, 11.5, 21.5, 0, "weed", 0.75),
    )
    assert (
        result.annotated_path
        == tmp_path / "onboard" / "annotated" / frame.metadata.filename
    )
    assert result.json_path == tmp_path / "onboard" / "detections" / (
        Path(frame.metadata.filename).stem + ".json"
    )
    assert result.annotated_path.is_file()
    with Image.open(result.annotated_path) as annotated:
        assert annotated.getpixel((0, 0)) == (30, 20, 10)

    record = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert record == {
        "frame_id": 1,
        "captured_at": "2026-10-06T12:00:00+00:00",
        "engine_path": str(engine_path.resolve()),
        "processed_path": str(frame.processed_path),
        "annotated_path": str(result.annotated_path),
        "detections": [
            {
                "x_min": 1.0,
                "y_min": 2.0,
                "x_max": 10.0,
                "y_max": 20.0,
                "class_id": 1,
                "class_name": "crop",
                "confidence": 0.9,
            },
            {
                "x_min": 3.5,
                "y_min": 4.5,
                "x_max": 11.5,
                "y_max": 21.5,
                "class_id": 0,
                "class_name": "weed",
                "confidence": 0.75,
            },
        ],
    }


@pytest.mark.parametrize("boxes", [None, FakeBoxes([], [], [])])
def test_infer_supports_frames_without_detections(
    monkeypatch, tmp_path: Path, boxes
) -> None:
    frame = acquired_frame(tmp_path)
    engine_path = tmp_path / "model.engine"
    engine_path.touch()
    install_fake_model(monkeypatch, FakeResult(boxes))

    result = InferencePipeline(engine_path, storage_root=tmp_path / "onboard").infer(
        frame
    )

    assert result.detections == ()
    assert json.loads(result.json_path.read_text(encoding="utf-8"))["detections"] == []


def test_inference_failure_keeps_acquired_images_without_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    frame = acquired_frame(tmp_path)
    engine_path = tmp_path / "model.engine"
    engine_path.touch()

    class FailingModel:
        def predict(self, **kwargs):
            raise RuntimeError("inference failed")

    monkeypatch.setattr(
        "onboard.inference.yolo_class",
        lambda: lambda engine_path, task: FailingModel(),
    )
    pipeline = InferencePipeline(engine_path, storage_root=tmp_path / "onboard")

    with pytest.raises(RuntimeError, match="inference failed"):
        pipeline.infer(frame)

    assert frame.raw_path.is_file()
    assert frame.processed_path.is_file()
    assert list(pipeline.annotated_dir.iterdir()) == []
    assert list(pipeline.detections_dir.iterdir()) == []

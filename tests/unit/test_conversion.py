from pathlib import Path
from typing import ClassVar

import pytest

from model_development.conversion import ModelArtifactConverter


class FakeExportModel:
    calls: ClassVar[list[dict]] = []

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = Path(checkpoint)

    def export(self, **kwargs) -> None:
        self.calls.append(kwargs)
        self.checkpoint.with_suffix(".onnx").write_bytes(b"onnx")
        self.checkpoint.with_suffix(".engine").write_bytes(b"engine")


def test_converter_exports_static_fp16_artifacts(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    FakeExportModel.calls.clear()
    monkeypatch.setattr(
        "model_development.conversion.yolo_class", lambda: FakeExportModel
    )

    result = ModelArtifactConverter(checkpoint, "1", 320).convert()

    assert result.onnx_path == tmp_path / "best.onnx"
    assert result.engine_path == tmp_path / "best.engine"
    assert FakeExportModel.calls == [
        {
            "format": "engine",
            "device": "1",
            "imgsz": 320,
            "batch": 1,
            "dynamic": False,
            "half": True,
        }
    ]


def test_converter_validates_checkpoint_and_device(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ModelArtifactConverter(tmp_path / "missing.pt", "0", 640)
    checkpoint = tmp_path / "best.onnx"
    checkpoint.touch()
    with pytest.raises(ValueError, match=r"\.pt"):
        ModelArtifactConverter(checkpoint, "0", 640)
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    with pytest.raises(ValueError, match="CUDA"):
        ModelArtifactConverter(checkpoint, "cpu", 640)


def test_converter_rejects_missing_export_outputs(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()

    class IncompleteExportModel:
        def __init__(self, checkpoint: str) -> None:
            pass

        def export(self, **kwargs) -> None:
            checkpoint.with_suffix(".onnx").touch()

    monkeypatch.setattr(
        "model_development.conversion.yolo_class", lambda: IncompleteExportModel
    )

    with pytest.raises(RuntimeError, match="best.engine"):
        ModelArtifactConverter(checkpoint, "0", 640).convert()

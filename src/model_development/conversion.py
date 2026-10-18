"""Convert trained YOLO checkpoints into deployment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from ultralytics import YOLO


@dataclass(frozen=True, slots=True)
class ConversionResult:
    onnx_path: Path
    engine_path: Path


class ModelArtifactConverter:
    """Export one best checkpoint as static FP16 ONNX and TensorRT artifacts."""

    def __init__(self, checkpoint: str | Path, device: str, imgsz: int) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if self.checkpoint.suffix.lower() != ".pt":
            raise ValueError("Export checkpoint must be a .pt file")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Export checkpoint does not exist: {self.checkpoint}")
        if not device.isdecimal():
            raise ValueError("TensorRT export requires a CUDA device index")
        self.device = device
        self.imgsz = imgsz

    def convert(self) -> ConversionResult:
        """Run Ultralytics' PT-to-ONNX-to-TensorRT export and verify its outputs."""
        model = YOLO(str(self.checkpoint))
        model.export(
            format="engine",
            device=self.device,
            imgsz=self.imgsz,
            batch=1,
            dynamic=False,
            half=True,
        )

        result = ConversionResult(
            onnx_path=self.checkpoint.with_suffix(".onnx"),
            engine_path=self.checkpoint.with_suffix(".engine"),
        )
        missing = [path.name for path in (result.onnx_path, result.engine_path) if not path.is_file()]
        if missing:
            raise RuntimeError(f"Model export did not create: {', '.join(missing)}")
        return result

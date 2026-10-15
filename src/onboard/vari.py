"""Compute and store per-detection VARI maps from processed RGB frames."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .acquisition import FrameMetadata
from .inference import Detection, InferenceResult


@dataclass(frozen=True, slots=True)
class DetectionVari:
    """A box's VARI pixels and statistics; bounds are exclusive at the maximum."""

    detection_index: int
    detection: Detection
    bounds: tuple[int, int, int, int]
    pixels: NDArray[np.float32]
    total_pixels: int
    valid_pixels: int


@dataclass(frozen=True, slots=True)
class VariResult:
    """Computed detections and stored outputs for one frame."""

    metadata: FrameMetadata
    processed_path: Path
    json_path: Path
    npz_path: Path
    detections: tuple[DetectionVari, ...]


class VariCalculator:
    """Reuse a storage directory while computing VARI for inference results."""

    def __init__(self, storage_root: str | Path = "data/onboard") -> None:
        self.vari_dir = Path(storage_root).expanduser().resolve() / "vari"
        self.vari_dir.mkdir(parents=True, exist_ok=True)

    def compute(self, result: InferenceResult) -> VariResult:
        """Compute all box pixels and replace this frame's JSON and NPZ outputs."""
        if not isinstance(result, InferenceResult):
            raise TypeError("VariCalculator expects an InferenceResult")
        if not result.processed_path.is_file():
            raise FileNotFoundError(
                f"Processed image does not exist: {result.processed_path}"
            )
        with Image.open(result.processed_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)

        detections = tuple(
            self._compute_detection(index, detection, rgb)
            for index, detection in enumerate(result.detections)
        )
        stem = Path(result.metadata.filename).stem
        output = VariResult(
            metadata=result.metadata,
            processed_path=result.processed_path,
            json_path=self.vari_dir / f"{stem}.json",
            npz_path=self.vari_dir / f"{stem}.npz",
            detections=detections,
        )
        self._save(output, result.json_path)
        return output

    @staticmethod
    def _compute_detection(
        index: int, detection: Detection, rgb: NDArray[np.float32]
    ) -> DetectionVari:
        coordinates = (
            detection.x_min, detection.y_min, detection.x_max, detection.y_max
        )
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"Detection {index} has nonfinite box coordinates")
        if detection.x_min > detection.x_max or detection.y_min > detection.y_max:
            raise ValueError(f"Detection {index} has reversed box coordinates")

        height, width = rgb.shape[:2]
        x_min = min(width, max(0, math.floor(detection.x_min)))
        y_min = min(height, max(0, math.floor(detection.y_min)))
        x_max = min(width, max(0, math.ceil(detection.x_max)))
        y_max = min(height, max(0, math.ceil(detection.y_max)))
        crop = rgb[y_min:y_max, x_min:x_max]
        red, green, blue = crop[..., 0], crop[..., 1], crop[..., 2]
        denominator = green + red - blue
        pixels = np.full(denominator.shape, np.nan, dtype=np.float32)
        np.divide(green - red, denominator, out=pixels, where=denominator != 0)
        return DetectionVari(
            detection_index=index,
            detection=detection,
            bounds=(x_min, y_min, x_max, y_max),
            pixels=pixels,
            total_pixels=int(pixels.size),
            valid_pixels=int(np.count_nonzero(np.isfinite(pixels))),
        )

    @staticmethod
    def _save(result: VariResult, inference_json_path: Path) -> None:
        arrays = {}
        records = []
        for detection in result.detections:
            key = f"detection_{detection.detection_index}"
            arrays[key] = detection.pixels
            records.append({
                "detection_index": detection.detection_index,
                "detection": asdict(detection.detection),
                "bounds": detection.bounds,
                "array_key": key,
                "total_pixels": detection.total_pixels,
                "valid_pixels": detection.valid_pixels,
            })
        record = {
            "frame_id": result.metadata.frame_id,
            "captured_at": result.metadata.captured_at.isoformat(),
            "processed_path": str(result.processed_path),
            "inference_json_path": str(inference_json_path),
            "npz_path": str(result.npz_path),
            "detections": records,
        }
        # Validate JSON before writing either output (NaN is only valid in NPZ).
        serialized = json.dumps(record, indent=2, allow_nan=False) + "\n"
        np.savez_compressed(result.npz_path, **arrays)
        result.json_path.write_text(serialized, encoding="utf-8")

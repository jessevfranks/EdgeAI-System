"""Run object detection on processed onboard frames."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

from .acquisition import AcquiredFrame, FrameMetadata


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object in processed-image pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Stored outputs and detections for one processed frame."""

    metadata: FrameMetadata
    engine_path: Path
    processed_path: Path
    annotated_path: Path
    json_path: Path
    detections: tuple[Detection, ...]


class InferencePipeline:
    """Load one TensorRT engine and reuse it for processed frame inference."""

    def __init__(
        self,
        engine_path: str | Path,
        device: str = "0",
        imgsz: int = 640,
        confidence: float = 0.25,
        iou: float = 0.7,
        storage_root: str | Path = "data/onboard",
    ) -> None:
        self.engine_path = Path(engine_path).expanduser().resolve()
        if self.engine_path.suffix.lower() != ".engine":
            raise ValueError("Inference model must be a TensorRT .engine file")
        if not self.engine_path.is_file():
            raise FileNotFoundError(
                f"TensorRT engine does not exist: {self.engine_path}"
            )

        self.device = device
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou

        storage_root = Path(storage_root).expanduser().resolve()
        self.annotated_dir = storage_root / "annotated"
        self.detections_dir = storage_root / "detections"
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.detections_dir.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(str(self.engine_path), task="detect")

    def infer(self, frame: AcquiredFrame) -> InferenceResult:
        """Infer from a stored processed image and persist its results."""
        if not isinstance(frame, AcquiredFrame):
            raise TypeError("InferencePipeline expects an AcquiredFrame")
        if not frame.processed_path.is_file():
            raise FileNotFoundError(
                f"Processed image does not exist: {frame.processed_path}"
            )

        results = self.model.predict(
            source=str(frame.processed_path),
            device=self.device,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
            save=False,
        )
        if len(results) != 1:
            raise RuntimeError("Expected one inference result for one processed frame")

        model_result = results[0]
        detections = self._detections(model_result)
        annotated_path = self.annotated_dir / frame.metadata.filename
        json_path = self.detections_dir / f"{Path(frame.metadata.filename).stem}.json"

        # Ultralytics returns plotted images in BGR channel order.
        plotted = model_result.plot()
        Image.fromarray(plotted[:, :, ::-1]).save(annotated_path, format="PNG")

        inference_result = InferenceResult(
            metadata=frame.metadata,
            engine_path=self.engine_path,
            processed_path=frame.processed_path,
            annotated_path=annotated_path,
            json_path=json_path,
            detections=detections,
        )
        self._save_json(inference_result)
        return inference_result

    @staticmethod
    def _detections(model_result: Any) -> tuple[Detection, ...]:
        boxes = model_result.boxes
        if boxes is None or len(boxes) == 0:
            return ()

        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()

        detections = []
        for coordinates_row, confidence, class_id_value in zip(
            coordinates, confidences, class_ids, strict=True
        ):
            class_id = int(class_id_value)
            detections.append(
                Detection(
                    x_min=float(coordinates_row[0]),
                    y_min=float(coordinates_row[1]),
                    x_max=float(coordinates_row[2]),
                    y_max=float(coordinates_row[3]),
                    class_id=class_id,
                    class_name=str(model_result.names[class_id]),
                    confidence=float(confidence),
                )
            )
        return tuple(detections)

    @staticmethod
    def _save_json(result: InferenceResult) -> None:
        record = {
            "frame_id": result.metadata.frame_id,
            "captured_at": result.metadata.captured_at.isoformat(),
            "engine_path": str(result.engine_path),
            "processed_path": str(result.processed_path),
            "annotated_path": str(result.annotated_path),
            "detections": [asdict(detection) for detection in result.detections],
        }
        result.json_path.write_text(
            json.dumps(record, indent=2) + "\n",
            encoding="utf-8",
        )

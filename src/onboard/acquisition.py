"""Receive, identify, process, and store onboard RGB images."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, PngImagePlugin

from .processing import ImagePipeline

FRAME_FILENAME_PATTERN = re.compile(r"^frame_(\d+)_\d{8}T\d{12}Z\.png$")


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    frame_id: int
    captured_at: datetime

    @property
    def filename(self) -> str:
        timestamp = self.captured_at.strftime("%Y%m%dT%H%M%S%fZ")
        return f"frame_{self.frame_id:06d}_{timestamp}.png"


@dataclass(frozen=True, slots=True)
class AcquiredFrame:
    metadata: FrameMetadata
    raw_path: Path
    processed_path: Path


class ImageAcquirer:
    """Store each received frame before running the processing pipeline."""

    def __init__(
        self,
        storage_root: str | Path = "data/onboard",
        pipeline: ImagePipeline | None = None,
    ) -> None:
        self.storage_root = Path(storage_root).expanduser().resolve()
        self.raw_dir = self.storage_root / "raw"
        self.processed_dir = self.storage_root / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = pipeline if pipeline is not None else ImagePipeline()
        self._next_frame_id = self._find_next_frame_id()

    def acquire(
        self,
        image: Image.Image,
        captured_at: datetime | None = None,
    ) -> AcquiredFrame:
        if not isinstance(image, Image.Image):
            raise TypeError("ImageAcquirer expects a Pillow image")

        captured_at = self._utc_timestamp(captured_at)
        metadata = FrameMetadata(self._next_frame_id, captured_at)
        self._next_frame_id += 1

        raw_path = self.raw_dir / metadata.filename
        processed_path = self.processed_dir / metadata.filename
        rgb_image = image.convert("RGB")

        self._save_png(rgb_image, raw_path, metadata)
        processed_image = self.pipeline.process(rgb_image.copy())
        self._save_png(processed_image, processed_path, metadata)

        return AcquiredFrame(metadata, raw_path, processed_path)

    def _find_next_frame_id(self) -> int:
        highest_id = 0
        for directory in (self.raw_dir, self.processed_dir):
            for path in directory.glob("frame_*.png"):
                match = FRAME_FILENAME_PATTERN.fullmatch(path.name)
                if match:
                    highest_id = max(highest_id, int(match.group(1)))
        return highest_id + 1

    @staticmethod
    def _utc_timestamp(value: datetime | None) -> datetime:
        if value is None:
            return datetime.now(UTC)
        if not isinstance(value, datetime):
            raise TypeError("Capture timestamp must be a datetime")
        if value.utcoffset() is None:
            raise ValueError("Capture timestamp must include a timezone")
        return value.astimezone(UTC)

    @staticmethod
    def _save_png(image: Image.Image, path: Path, metadata: FrameMetadata) -> None:
        png_metadata = PngImagePlugin.PngInfo()
        png_metadata.add_text("frame_id", str(metadata.frame_id))
        png_metadata.add_text("captured_at", metadata.captured_at.isoformat())
        image.save(path, format="PNG", pnginfo=png_metadata)

"""Ordered image-processing pipeline for onboard frames."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, PngImagePlugin

from .acquisition import AcquiredFrame, FrameMetadata

ImageProcessor = Callable[[Image.Image], Image.Image]


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    metadata: FrameMetadata
    processed_path: Path


class ImagePipeline:
    """Process acquired frames and store the resulting images."""

    def __init__(
        self,
        processors: Iterable[ImageProcessor] = (),
        storage_root: str | Path = "data/onboard",
    ) -> None:
        self._processors: list[ImageProcessor] = []
        for processor in processors:
            self.add(processor)
        self.processed_dir = Path(storage_root).expanduser().resolve() / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def add(self, processor: ImageProcessor) -> None:
        if not callable(processor):
            raise TypeError("Image processor must be callable")
        self._processors.append(processor)

    def process(self, frame: AcquiredFrame) -> ProcessedFrame:
        if not isinstance(frame, AcquiredFrame):
            raise TypeError("ImagePipeline expects an AcquiredFrame")
        if not frame.raw_path.is_file():
            raise FileNotFoundError(f"Raw image does not exist: {frame.raw_path}")

        with Image.open(frame.raw_path) as image:
            result = image.convert("RGB")
            for processor in self._processors:
                result = processor(result)
                if not isinstance(result, Image.Image):
                    raise TypeError("Image processors must return a Pillow image")

            processed_path = self.processed_dir / frame.metadata.filename
            png_metadata = PngImagePlugin.PngInfo()
            png_metadata.add_text("frame_id", str(frame.metadata.frame_id))
            png_metadata.add_text("captured_at", frame.metadata.captured_at.isoformat())
            result.convert("RGB").save(
                processed_path, format="PNG", pnginfo=png_metadata
            )

        return ProcessedFrame(frame.metadata, processed_path)

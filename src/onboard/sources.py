"""Test image sources for the onboard pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".avif", ".webp"}
_open_image = Image.open
if _open_image.__module__ == "ultralytics.utils.patches":
    # Ultralytics wraps all Pillow failures with an optional HEIF install attempt.
    # These sources do not support HEIF, so retain Pillow's normal error behavior.
    from ultralytics.utils.patches import _image_open as _open_image


@dataclass(frozen=True, slots=True)
class StreamFrame:
    """One image and its source information before onboard acquisition."""

    image: Image.Image
    captured_at: datetime
    source_path: Path


class PacedImageDirectory:
    """Yield each supported image in a directory once at a fixed rate."""

    def __init__(self, directory: str | Path, fps: float) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if not self.directory.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.directory}")
        if not self.directory.is_dir():
            raise ValueError(f"Image source must be a directory: {self.directory}")
        if fps <= 0:
            raise ValueError("FPS must be greater than zero")

        self.fps = float(fps)
        self.image_paths = tuple(
            path
            for path in self.directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        if not self.image_paths:
            raise ValueError(f"Image directory contains no supported images: {self.directory}")

    def __iter__(self) -> Iterator[StreamFrame]:
        started_at = time.monotonic()
        interval = 1.0 / self.fps

        for index, path in enumerate(self.image_paths):
            try:
                with _open_image(path) as image:
                    loaded_image = image.copy()  # TODO: investigate efficiency surrounding copying images around/having different objects at each step
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"Could not read image: {path}") from error

            delay = started_at + index * interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            yield StreamFrame(
                image=loaded_image,
                captured_at=datetime.now(UTC),
                source_path=path,
            )

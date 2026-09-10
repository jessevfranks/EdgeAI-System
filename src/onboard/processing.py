"""Ordered image-processing pipeline for onboard frames."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from PIL import Image

ImageProcessor = Callable[[Image.Image], Image.Image]


class ImagePipeline:
    """Apply image processors in the order in which they were added."""

    def __init__(self, processors: Iterable[ImageProcessor] = ()) -> None:
        self._processors: list[ImageProcessor] = []
        for processor in processors:
            self.add(processor)

    def add(self, processor: ImageProcessor) -> None:
        if not callable(processor):
            raise TypeError("Image processor must be callable")
        self._processors.append(processor)

    def process(self, image: Image.Image) -> Image.Image:
        result = image
        for processor in self._processors:
            result = processor(result)
            if not isinstance(result, Image.Image):
                raise TypeError("Image processors must return a Pillow image")
        return result

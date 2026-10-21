"""Run the onboard image pipeline from acquisition through VARI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image

from .acquisition import AcquiredFrame, ImageAcquirer
from .inference import InferencePipeline, InferenceResult
from .processing import ImagePipeline, ProcessedFrame
from .sources import StreamFrame
from .vari import VariCalculator, VariResult


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    acquired_frame: AcquiredFrame
    processed_frame: ProcessedFrame
    inference_result: InferenceResult
    vari_result: VariResult


class OnboardOrchestrator:
    """Construct and run each stage of the onboard image pipeline."""

    def __init__(
        self,
        model_path: str | Path,
        storage_root: str | Path = "data/onboard",
        device: str = "0",
        imgsz: int = 640,
        confidence: float = 0.25,
        iou: float = 0.7,
    ) -> None:
        self.acquirer = ImageAcquirer(storage_root)
        self.processor = ImagePipeline(storage_root=storage_root)
        # TODO: Add image-processing steps to self.processor here.
        self.inference = InferencePipeline(
            model_path=model_path,
            device=device,
            imgsz=imgsz,
            confidence=confidence,
            iou=iou,
            storage_root=storage_root,
        )
        self.vari = VariCalculator(storage_root)

        # TODO: Run independent frames concurrently when the pipeline requires it.
        # TODO: Add metrics collection and logging.

    def run(
        self,
        image: Image.Image,
        captured_at: datetime | None = None,
    ) -> OrchestrationResult:
        acquired_frame = self.acquirer.acquire(image, captured_at)
        processed_frame = self.processor.process(acquired_frame)
        inference_result = self.inference.infer(processed_frame)
        vari_result = self.vari.compute(inference_result)
        return OrchestrationResult(
            acquired_frame=acquired_frame,
            processed_frame=processed_frame,
            inference_result=inference_result,
            vari_result=vari_result,
        )

    def run_stream(
        self,
        frames: Iterable[StreamFrame],
    ) -> Iterator[OrchestrationResult]:
        """Process input frames sequentially until the source is exhausted."""
        for frame in frames:
            yield self.run(frame.image, frame.captured_at)

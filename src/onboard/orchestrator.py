"""Run the onboard image pipeline from acquisition through VARI."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from .acquisition import AcquiredFrame, ImageAcquirer
from .inference import InferencePipeline, InferenceResult
from .metrics import MetricsCollector
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
        metrics: MetricsCollector | None = None,
    ) -> None:
        self.metrics = metrics
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
            metrics=metrics,
        )
        self.vari = VariCalculator(storage_root)

        # TODO: Run independent frames concurrently when the pipeline requires it.

    def run(
        self,
        image: Image.Image,
        captured_at: datetime | None = None,
    ) -> OrchestrationResult:
        if self.metrics is None:
            return self._run_frame(image, captured_at)
        with self.metrics.run():
            return self._run_frame(image, captured_at)

    def _run_frame(
        self,
        image: Image.Image,
        captured_at: datetime | None,
    ) -> OrchestrationResult:
        pipeline_started = time.perf_counter()
        acquired_frame = self.acquirer.acquire(image, captured_at)
        processed_frame = self.processor.process(acquired_frame)
        inference_started = time.perf_counter()
        inference_result = self.inference.infer(processed_frame)
        inference_latency_ms = (time.perf_counter() - inference_started) * 1000
        vari_result = self.vari.compute(inference_result)
        result = OrchestrationResult(
            acquired_frame=acquired_frame,
            processed_frame=processed_frame,
            inference_result=inference_result,
            vari_result=vari_result,
        )
        if self.metrics is not None:
            self.metrics.record_frame(
                frame_id=acquired_frame.metadata.frame_id,
                pipeline_latency_ms=(time.perf_counter() - pipeline_started) * 1000,
                inference_stage_latency_ms=inference_latency_ms,
            )
        return result

    def run_stream(
        self,
        frames: Iterable[StreamFrame],
    ) -> Iterator[OrchestrationResult]:
        """Process input frames sequentially until the source is exhausted."""
        if self.metrics is None:
            for frame in frames:
                yield self._run_frame(frame.image, frame.captured_at)
            return

        with self.metrics.run():
            for frame in frames:
                yield self._run_frame(frame.image, frame.captured_at)

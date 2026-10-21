"""Onboard RGB image acquisition and processing."""

from .acquisition import AcquiredFrame, FrameMetadata, ImageAcquirer
from .config import OnboardConfig
from .sources import PacedImageDirectory, StreamFrame
from .inference import Detection, InferencePipeline, InferenceResult
from .orchestrator import OnboardOrchestrator, OrchestrationResult
from .processing import ImagePipeline, ImageProcessor, ProcessedFrame
from .vari import DetectionVari, VariCalculator, VariResult

__all__ = [
    "AcquiredFrame",
    "Detection",
    "DetectionVari",
    "FrameMetadata",
    "ImageAcquirer",
    "ImagePipeline",
    "ImageProcessor",
    "InferencePipeline",
    "InferenceResult",
    "OnboardOrchestrator",
    "OnboardConfig",
    "OrchestrationResult",
    "PacedImageDirectory",
    "ProcessedFrame",
    "StreamFrame",
    "VariCalculator",
    "VariResult",
]

"""Onboard RGB image acquisition and processing."""

from .acquisition import AcquiredFrame, FrameMetadata, ImageAcquirer
from .config import OnboardConfig
from .inference import Detection, InferencePipeline, InferenceResult
from .metrics import MetricsCollector
from .orchestrator import OnboardOrchestrator, OrchestrationResult
from .processing import ImagePipeline, ImageProcessor, ProcessedFrame
from .sources import PacedImageDirectory, StreamFrame
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
    "MetricsCollector",
    "OnboardConfig",
    "OnboardOrchestrator",
    "OrchestrationResult",
    "PacedImageDirectory",
    "ProcessedFrame",
    "StreamFrame",
    "VariCalculator",
    "VariResult",
]

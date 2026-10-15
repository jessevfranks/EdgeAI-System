"""Onboard RGB image acquisition and processing."""

from .acquisition import AcquiredFrame, FrameMetadata, ImageAcquirer
from .inference import Detection, InferencePipeline, InferenceResult
from .processing import ImagePipeline, ImageProcessor
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
    "VariCalculator",
    "VariResult",
]

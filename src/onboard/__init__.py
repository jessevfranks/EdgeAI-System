"""Onboard RGB image acquisition and processing."""

from .acquisition import AcquiredFrame, FrameMetadata, ImageAcquirer
from .inference import Detection, InferencePipeline, InferenceResult
from .processing import ImagePipeline, ImageProcessor

__all__ = [
    "AcquiredFrame",
    "Detection",
    "FrameMetadata",
    "ImageAcquirer",
    "ImagePipeline",
    "ImageProcessor",
    "InferencePipeline",
    "InferenceResult",
]

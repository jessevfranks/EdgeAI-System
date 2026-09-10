"""Onboard RGB image acquisition and processing."""

from .acquisition import AcquiredFrame, FrameMetadata, ImageAcquirer
from .processing import ImagePipeline, ImageProcessor

__all__ = [
    "AcquiredFrame",
    "FrameMetadata",
    "ImageAcquirer",
    "ImagePipeline",
    "ImageProcessor",
]

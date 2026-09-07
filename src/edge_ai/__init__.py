"""EdgeAI research tooling for YOLOv8 training and onboard inference."""

from .config import ExperimentConfig, ModelScale
from .storage import ExperimentRecord, ExperimentStorage

__all__ = [
    "ExperimentConfig",
    "ExperimentRecord",
    "ExperimentStorage",
    "ModelScale",
]

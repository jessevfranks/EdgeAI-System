"""Training workflow adapter used by the background worker."""

from edge_ai.experiments import promote_best, train_model

__all__ = ["promote_best", "train_model"]

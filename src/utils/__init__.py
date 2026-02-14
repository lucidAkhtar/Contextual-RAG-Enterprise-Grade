"""Utilities package for logging, metrics, and helper functions."""

from .logger import setup_logger
from .metrics import (
    MetricsCollector,
    measure_time,
    timed,
    cosine_similarity,
    calculate_recall_at_k
)

__all__ = [
    "setup_logger",
    "MetricsCollector",
    "measure_time",
    "timed",
    "cosine_similarity",
    "calculate_recall_at_k"
]

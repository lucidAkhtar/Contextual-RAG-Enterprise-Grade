"""Metrics and evaluation utilities."""

import time
import numpy as np
from typing import List, Dict, Any
from functools import wraps
from contextlib import contextmanager


class MetricsCollector:
    """
    Collects and aggregates performance metrics.
    Implements the Observer pattern for metric collection.
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        values = np.array(self.metrics[metric_name])
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "count": len(values)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics.keys()}
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()


@contextmanager
def measure_time(metrics_collector: MetricsCollector, metric_name: str):
    """
    Context manager to measure execution time.
    
    Usage:
        with measure_time(collector, "query_latency"):
            # code to measure
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        metrics_collector.record(metric_name, elapsed)


def timed(metric_name: str):
    """
    Decorator to measure function execution time.
    
    Usage:
        @timed("function_latency")
        def my_function():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            # Store in function attribute for retrieval
            if not hasattr(wrapper, 'latencies'):
                wrapper.latencies = []
            wrapper.latencies.append(elapsed)
            return result
        return wrapper
    return decorator


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    if norm_product == 0:
        return 0.0
    
    return float(dot_product / norm_product)


def calculate_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Calculate Recall@K metric.
    
    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k: Top-K to consider
    
    Returns:
        Recall@K score
    """
    if not relevant_ids:
        return 0.0
    
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    relevant_retrieved = top_k_retrieved & relevant_set
    
    return len(relevant_retrieved) / len(relevant_set)

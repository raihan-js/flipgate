"""Statistical analysis module for FlipGate."""

from .mcnemar import mcnemar_test, count_flips
from .bootstrap import paired_bootstrap_ci, compute_flip_rate, compare_to_floor

__all__ = [
    "mcnemar_test", "count_flips",
    "paired_bootstrap_ci", "compute_flip_rate", "compare_to_floor",
]

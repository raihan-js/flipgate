"""Paired bootstrap confidence intervals for flip rates."""

from typing import NamedTuple
import numpy as np


class BootstrapResult(NamedTuple):
    """Result of paired bootstrap analysis."""
    estimate: float  # Point estimate (mean)
    ci_lower: float  # Lower bound of CI
    ci_upper: float  # Upper bound of CI
    std_error: float  # Standard error


def compute_flip_rate(
    baseline_correct: np.ndarray | list[bool],
    candidate_correct: np.ndarray | list[bool],
    direction: str = "right_to_wrong",
) -> float:
    """Compute flip rate between baseline and candidate.
    
    Args:
        baseline_correct: Binary array, True if baseline got item correct
        candidate_correct: Binary array, True if candidate got item correct
        direction: "right_to_wrong", "wrong_to_right", or "any"
    
    Returns:
        Flip rate as a proportion (0.0 to 1.0)
    """
    baseline = np.asarray(baseline_correct, dtype=bool)
    candidate = np.asarray(candidate_correct, dtype=bool)
    n = len(baseline)
    
    if n == 0:
        return 0.0
    
    if direction == "right_to_wrong":
        flips = np.sum(baseline & ~candidate)
    elif direction == "wrong_to_right":
        flips = np.sum(~baseline & candidate)
    elif direction == "any":
        flips = np.sum(baseline != candidate)
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    return float(flips / n)


def paired_bootstrap_ci(
    baseline_correct: np.ndarray | list[bool],
    candidate_correct: np.ndarray | list[bool],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    direction: str = "right_to_wrong",
    seed: int | None = None,
) -> BootstrapResult:
    """Compute paired bootstrap confidence interval for flip rate.
    
    Uses the percentile method to compute confidence intervals for the
    difference in flip rates between baseline and candidate.
    
    Args:
        baseline_correct: Binary array for baseline
        candidate_correct: Binary array for candidate
        n_bootstrap: Number of bootstrap samples (default 1000)
        confidence_level: Confidence level for CI (default 0.95 for 95% CI)
        direction: "right_to_wrong", "wrong_to_right", or "any"
        seed: Random seed for reproducibility
    
    Returns:
        BootstrapResult with point estimate, CI bounds, and standard error
    
    Example:
        >>> baseline = [True, True, False, True, False, True, True, False]
        >>> candidate = [True, False, False, True, True, True, False, False]
        >>> result = paired_bootstrap_ci(baseline, candidate)
        >>> print(f"Flip rate: {result.estimate:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    """
    baseline = np.asarray(baseline_correct, dtype=bool)
    candidate = np.asarray(candidate_correct, dtype=bool)
    
    if len(baseline) != len(candidate):
        raise ValueError("Arrays must have the same length")
    
    n = len(baseline)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, 0.0)
    
    rng = np.random.default_rng(seed)
    
    # Bootstrap sampling
    bootstrap_estimates = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = rng.integers(0, n, size=n)
        boot_baseline = baseline[indices]
        boot_candidate = candidate[indices]
        
        bootstrap_estimates[i] = compute_flip_rate(
            boot_baseline, boot_candidate, direction=direction
        )
    
    # Compute statistics
    point_estimate = compute_flip_rate(baseline, candidate, direction=direction)
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_estimates, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))
    std_error = float(np.std(bootstrap_estimates, ddof=1))
    
    return BootstrapResult(
        estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
    )


def compare_to_floor(
    candidate_flip_rate: float,
    floor_estimate: float,
    floor_ci_upper: float,
    margin: float = 1.0,
) -> dict:
    """Compare candidate flip rate to noise floor.
    
    A candidate fails the gate if its flip rate exceeds the noise floor
    plus a margin. The margin accounts for the uncertainty in the floor
    estimate.
    
    Args:
        candidate_flip_rate: Observed flip rate for candidate
        floor_estimate: Point estimate of noise floor
        floor_ci_upper: Upper bound of floor CI
        margin: Multiplier for floor (default 1.0 = fail if above floor)
    
    Returns:
        Dictionary with:
        - passes: bool, whether candidate passes the gate
        - ratio: candidate flip rate / floor estimate
        - exceeds_by: how much candidate exceeds floor (in percentage points)
    """
    threshold = floor_ci_upper * margin
    passes = candidate_flip_rate <= threshold
    
    if floor_estimate > 0:
        ratio = candidate_flip_rate / floor_estimate
    else:
        ratio = float('inf') if candidate_flip_rate > 0 else 1.0
    
    exceeds_by = candidate_flip_rate - floor_estimate
    
    return {
        "passes": passes,
        "ratio": ratio,
        "exceeds_by": exceeds_by,
        "threshold": threshold,
    }

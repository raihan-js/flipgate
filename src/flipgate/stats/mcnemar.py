"""McNemar's test for paired nominal data."""

from typing import NamedTuple
import numpy as np
from scipy import stats


class McNemarResult(NamedTuple):
    """Result of McNemar's test."""
    statistic: float
    p_value: float
    n_01: int  # baseline correct, candidate wrong
    n_10: int  # baseline wrong, candidate correct
    significant: bool  # at alpha=0.05


def mcnemar_test(
    baseline_correct: np.ndarray | list[bool],
    candidate_correct: np.ndarray | list[bool],
    alpha: float = 0.05,
    exact: bool = False,
) -> McNemarResult:
    """Perform McNemar's test on paired binary outcomes.
    
    Tests whether the marginal probabilities of two paired binary outcomes
    are equal. In FlipGate's context: does the candidate have a different
    error rate than the baseline?
    
    Args:
        baseline_correct: Binary array, True if baseline got item correct
        candidate_correct: Binary array, True if candidate got item correct
        alpha: Significance level (default 0.05)
        exact: Use exact binomial test instead of chi-squared approximation
    
    Returns:
        McNemarResult with test statistic, p-value, discordant pair counts,
        and whether the result is significant.
    
    Example:
        >>> baseline = [True, True, False, True, False]
        >>> candidate = [True, False, False, True, True]
        >>> result = mcnemar_test(baseline, candidate)
        >>> result.n_01  # baseline right, candidate wrong
        1
        >>> result.n_10  # baseline wrong, candidate right
        1
    """
    baseline = np.asarray(baseline_correct, dtype=bool)
    candidate = np.asarray(candidate_correct, dtype=bool)
    
    if len(baseline) != len(candidate):
        raise ValueError("Arrays must have the same length")
    
    # Count discordant pairs
    # n_01: baseline correct (1), candidate wrong (0) - right-to-wrong flips
    # n_10: baseline wrong (0), candidate correct (1) - wrong-to-right flips
    n_01 = int(np.sum(baseline & ~candidate))  # right-to-wrong
    n_10 = int(np.sum(~baseline & candidate))  # wrong-to-right
    
    if exact:
        # Exact binomial test
        # Under H0, n_01 and n_10 should be equally likely
        n_total = n_01 + n_10
        if n_total == 0:
            return McNemarResult(0.0, 1.0, n_01, n_10, False)
        # Two-sided exact binomial test (scipy >= 1.7 uses binomtest)
        bt = stats.binomtest(n_01, n_total, 0.5)
        p_value = bt.pvalue
        statistic = n_01  # For exact test, report the count
    else:
        # Chi-squared approximation with continuity correction
        if n_01 + n_10 == 0:
            return McNemarResult(0.0, 1.0, n_01, n_10, False)
        
        # McNemar's statistic with continuity correction
        statistic = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
        p_value = stats.chi2.sf(statistic, df=1)
    
    return McNemarResult(
        statistic=statistic,
        p_value=p_value,
        n_01=n_01,
        n_10=n_10,
        significant=p_value < alpha,
    )


def count_flips(
    baseline_correct: np.ndarray | list[bool],
    candidate_correct: np.ndarray | list[bool],
) -> dict[str, int]:
    """Count all flip types between baseline and candidate.
    
    Returns:
        Dictionary with keys:
        - right_to_wrong: baseline correct, candidate wrong
        - wrong_to_right: baseline wrong, candidate correct
        - both_correct: both got it right
        - both_wrong: both got it wrong
        - total_flips: right_to_wrong + wrong_to_right
    """
    baseline = np.asarray(baseline_correct, dtype=bool)
    candidate = np.asarray(candidate_correct, dtype=bool)
    
    right_to_wrong = int(np.sum(baseline & ~candidate))
    wrong_to_right = int(np.sum(~baseline & candidate))
    both_correct = int(np.sum(baseline & candidate))
    both_wrong = int(np.sum(~baseline & ~candidate))
    
    return {
        "right_to_wrong": right_to_wrong,
        "wrong_to_right": wrong_to_right,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "total_flips": right_to_wrong + wrong_to_right,
    }

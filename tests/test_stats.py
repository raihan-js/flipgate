"""Tests for statistical module."""

import pytest
import numpy as np
from flipgate.stats.mcnemar import mcnemar_test, count_flips
from flipgate.stats.bootstrap import (
    compute_flip_rate,
    paired_bootstrap_ci,
    compare_to_floor,
)


class TestMcNemar:
    def test_no_difference(self):
        baseline = [True, True, False, False]
        candidate = [True, True, False, False]
        result = mcnemar_test(baseline, candidate)
        assert result.p_value == 1.0
        assert not result.significant
    
    def test_symmetric_flips(self):
        # Equal number of right-to-wrong and wrong-to-right
        baseline = [True, True, False, False]
        candidate = [True, False, False, True]
        result = mcnemar_test(baseline, candidate)
        assert result.n_01 == 1  # right-to-wrong
        assert result.n_10 == 1  # wrong-to-right
        # With symmetric flips, should not be significant
        assert not result.significant
    
    def test_asymmetric_flips_significant(self):
        # Many more right-to-wrong than wrong-to-right
        baseline = [True] * 100 + [False] * 100
        candidate = [True] * 60 + [False] * 40 + [True] * 10 + [False] * 90
        result = mcnemar_test(baseline, candidate)
        assert result.n_01 == 40  # right-to-wrong
        assert result.n_10 == 10  # wrong-to-right
        # Should be significant with this asymmetry
        assert result.p_value < 0.05
    
    def test_exact_test(self):
        baseline = [True, True, False, False, True]
        candidate = [True, False, False, True, True]
        result = mcnemar_test(baseline, candidate, exact=True)
        assert result.n_01 == 1
        assert result.n_10 == 1
    
    def test_all_correct(self):
        baseline = [True, True, True]
        candidate = [True, True, True]
        result = mcnemar_test(baseline, candidate)
        assert result.p_value == 1.0
    
    def test_all_wrong(self):
        baseline = [False, False, False]
        candidate = [False, False, False]
        result = mcnemar_test(baseline, candidate)
        assert result.p_value == 1.0
    
    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            mcnemar_test([True, False], [True])


class TestCountFlips:
    def test_count_all_types(self):
        baseline = [True, True, False, False]
        candidate = [True, False, False, True]
        flips = count_flips(baseline, candidate)
        
        assert flips["right_to_wrong"] == 1
        assert flips["wrong_to_right"] == 1
        assert flips["both_correct"] == 1
        assert flips["both_wrong"] == 1
        assert flips["total_flips"] == 2
    
    def test_no_flips(self):
        baseline = [True, False, True, False]
        candidate = [True, False, True, False]
        flips = count_flips(baseline, candidate)
        
        assert flips["total_flips"] == 0
        assert flips["both_correct"] == 2
        assert flips["both_wrong"] == 2


class TestComputeFlipRate:
    def test_right_to_wrong_rate(self):
        baseline = [True, True, True, True]
        candidate = [True, False, True, False]
        rate = compute_flip_rate(baseline, candidate, direction="right_to_wrong")
        assert rate == 0.5  # 2 out of 4
    
    def test_wrong_to_right_rate(self):
        baseline = [False, False, False, False]
        candidate = [True, False, True, False]
        rate = compute_flip_rate(baseline, candidate, direction="wrong_to_right")
        assert rate == 0.5
    
    def test_any_flip_rate(self):
        baseline = [True, True, False, False]
        candidate = [True, False, False, True]
        rate = compute_flip_rate(baseline, candidate, direction="any")
        assert rate == 0.5  # 2 flips out of 4
    
    def test_empty_arrays(self):
        rate = compute_flip_rate([], [], direction="right_to_wrong")
        assert rate == 0.0
    
    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="Unknown direction"):
            compute_flip_rate([True], [False], direction="invalid")


class TestPairedBootstrap:
    def test_bootstrap_ci_basic(self):
        np.random.seed(42)
        baseline = [True] * 50 + [False] * 50
        candidate = [True] * 40 + [False] * 10 + [True] * 5 + [False] * 45
        
        result = paired_bootstrap_ci(baseline, candidate, n_bootstrap=100, seed=42)
        
        assert 0.0 <= result.ci_lower <= result.estimate <= result.ci_upper <= 1.0
        assert result.std_error >= 0
    
    def test_bootstrap_ci_no_flips(self):
        baseline = [True, True, False, False]
        candidate = [True, True, False, False]
        
        result = paired_bootstrap_ci(baseline, candidate, n_bootstrap=100, seed=42)
        
        assert result.estimate == 0.0
        assert result.ci_lower == 0.0
        assert result.ci_upper == 0.0
    
    def test_bootstrap_ci_empty(self):
        result = paired_bootstrap_ci([], [], n_bootstrap=100)
        assert result.estimate == 0.0
    
    def test_bootstrap_reproducibility(self):
        baseline = [True, False, True, False, True]
        candidate = [True, True, False, False, True]
        
        result1 = paired_bootstrap_ci(baseline, candidate, n_bootstrap=100, seed=42)
        result2 = paired_bootstrap_ci(baseline, candidate, n_bootstrap=100, seed=42)
        
        assert result1.estimate == result2.estimate
        assert result1.ci_lower == result2.ci_lower


class TestCompareToFloor:
    def test_passes_when_below_floor(self):
        result = compare_to_floor(
            candidate_flip_rate=0.02,
            floor_estimate=0.03,
            floor_ci_upper=0.04,
            margin=1.0,
        )
        assert result["passes"] is True
        assert result["ratio"] < 1.0
    
    def test_fails_when_above_floor(self):
        result = compare_to_floor(
            candidate_flip_rate=0.10,
            floor_estimate=0.03,
            floor_ci_upper=0.04,
            margin=1.0,
        )
        assert result["passes"] is False
        assert result["ratio"] > 1.0
    
    def test_margin_effect(self):
        # With margin=2.0, threshold is 2x the floor CI upper
        result = compare_to_floor(
            candidate_flip_rate=0.07,
            floor_estimate=0.03,
            floor_ci_upper=0.04,
            margin=2.0,  # threshold = 0.08
        )
        assert result["passes"] is True
    
    def test_zero_floor(self):
        result = compare_to_floor(
            candidate_flip_rate=0.05,
            floor_estimate=0.0,
            floor_ci_upper=0.0,
            margin=1.0,
        )
        assert result["passes"] is False
        assert result["ratio"] == float('inf')

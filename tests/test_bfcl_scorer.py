"""Tests for BFCL scorer (spec mode + exact gold mode)."""

import pytest
from flipgate.scorers.bfcl import BFCLScorer


@pytest.fixture
def scorer():
    return BFCLScorer()


def spec_ref():
    return {"function": [{
        "name": "calculate_triangle_area",
        "parameters": {"properties": {
            "base": {"type": "integer"},
            "height": {"type": "integer"},
        }},
    }]}


class TestSpecMode:
    def test_perfect_call(self, scorer):
        resp = '{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}'
        assert scorer.score("", resp, spec_ref()) == 1.0

    def test_wrong_name(self, scorer):
        resp = '{"name": "other_function", "arguments": {"base": 10, "height": 5}}'
        assert scorer.score("", resp, spec_ref()) == 0.0

    def test_missing_arg(self, scorer):
        # name match (0.5) + half args (0.5 * 1/2) = 0.75
        resp = '{"name": "calculate_triangle_area", "arguments": {"base": 10}}'
        assert scorer.score("", resp, spec_ref()) == 0.75

    def test_no_call_found(self, scorer):
        assert scorer.score("", "I don't know.", spec_ref()) == 0.0

    def test_is_correct(self, scorer):
        resp = '{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}'
        assert scorer.is_correct("", resp, spec_ref()) is True
        assert scorer.is_correct("", "nope", spec_ref()) is False


class TestExactMode:
    def test_exact_match(self, scorer):
        ref = {"ground_truth": ["calc_binomial_probability(n=20, k=5, p=0.6)"]}
        resp = '{"name": "calc_binomial_probability", "arguments": {"n": 20, "k": 5, "p": 0.6}}'
        assert scorer.score("", resp, ref) == 1.0

    def test_fraction_tolerance(self, scorer):
        ref = {"ground_truth": ["calc_binomial_probability(n=20, k=5, p=1/6)"]}
        resp = '{"name": "calc_binomial_probability", "arguments": {"n": 20, "k": 5, "p": 0.1666667}}'
        assert scorer.score("", resp, ref) == 1.0

    def test_wrong_value(self, scorer):
        ref = {"ground_truth": ["calc_binomial_probability(n=20, k=5, p=0.6)"]}
        resp = '{"name": "calc_binomial_probability", "arguments": {"n": 20, "k": 5, "p": 0.5}}'
        assert scorer.score("", resp, ref) == 0.0

    def test_wrong_name(self, scorer):
        ref = {"ground_truth": ["calc_binomial_probability(n=20, k=5, p=0.6)"]}
        resp = '{"name": "other_fn", "arguments": {"n": 20, "k": 5, "p": 0.6}}'
        assert scorer.score("", resp, ref) == 0.0

    def test_is_correct(self, scorer):
        ref = {"ground_truth": ["f(x=1)"]}
        assert scorer.is_correct("", '{"name": "f", "arguments": {"x": 1}}', ref) is True

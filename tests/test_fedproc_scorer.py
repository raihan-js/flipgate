"""Tests for FedProc registry scorer."""

import pytest
from flipgate.scorers.fedproc import FedProcRegistryScorer


@pytest.fixture
def scorer():
    registry = {"52.203-1", "52.212-4", "252.203-1", "52.219-1"}
    return FedProcRegistryScorer(registry=registry)


class TestFedProcRegistryScorer:
    def test_extract_clause_numbers_far(self, scorer):
        response = "According to FAR 52.203-1, contractors must..."
        clauses = scorer.extract_clause_numbers(response)
        assert clauses == ["52.203-1"]
    
    def test_extract_clause_numbers_dfars(self, scorer):
        response = "DFARS 252.203-1 requires compliance with..."
        clauses = scorer.extract_clause_numbers(response)
        assert clauses == ["252.203-1"]
    
    def test_extract_multiple_clauses(self, scorer):
        response = "See FAR 52.203-1 and FAR 52.212-4 for details."
        clauses = scorer.extract_clause_numbers(response)
        assert set(clauses) == {"52.203-1", "52.212-4"}
    
    def test_extract_no_clauses(self, scorer):
        response = "This is a general response without clause numbers."
        clauses = scorer.extract_clause_numbers(response)
        assert clauses == []
    
    def test_score_valid_clause(self, scorer):
        response = "Per FAR 52.203-1, the contractor shall..."
        assert scorer.score("", response, {}) == 1.0
    
    def test_score_hallucinated_clause(self, scorer):
        response = "According to FAR 99.999-9, you must..."
        assert scorer.score("", response, {}) == 0.0
    
    def test_score_mixed_valid_and_invalid(self, scorer):
        response = "FAR 52.203-1 is valid but FAR 99.999-9 is not."
        assert scorer.score("", response, {}) == 0.0
    
    def test_score_no_clauses_cited(self, scorer):
        response = "This is a general response."
        assert scorer.score("", response, {}) == 1.0
    
    def test_get_hallucinated_clauses(self, scorer):
        response = "FAR 52.203-1 is real, but FAR 99.999-9 is fake."
        hallucinated = scorer.get_hallucinated_clauses(response)
        assert hallucinated == ["99.999-9"]
    
    def test_get_hallucinated_clauses_none(self, scorer):
        response = "FAR 52.203-1 and FAR 52.212-4 are both valid."
        hallucinated = scorer.get_hallucinated_clauses(response)
        assert hallucinated == []
    
    def test_is_correct_with_valid_clauses(self, scorer):
        response = "Per FAR 52.203-1..."
        assert scorer.is_correct("", response, {}) is True
    
    def test_is_correct_with_invalid_clauses(self, scorer):
        response = "Per FAR 99.999-9..."
        assert scorer.is_correct("", response, {}) is False

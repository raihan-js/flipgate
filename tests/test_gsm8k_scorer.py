"""Tests for GSM8K scorer."""

import pytest
from flipgate.scorers.gsm8k import GSM8KScorer


@pytest.fixture
def scorer():
    return GSM8KScorer()


class TestGSM8KScorer:
    def test_extract_answer_with_hash_marker(self, scorer):
        response = "Let me solve this step by step.\n5 + 3 = 8\n#### 8"
        assert scorer.extract_answer(response) == 8.0
    
    def test_extract_answer_with_text(self, scorer):
        response = "After calculating, the answer is 42."
        assert scorer.extract_answer(response) == 42.0
    
    def test_extract_answer_with_dollar(self, scorer):
        response = "The total cost is $150."
        assert scorer.extract_answer(response) == 150.0
    
    def test_extract_answer_decimal(self, scorer):
        response = "The answer is 3.14."
        assert scorer.extract_answer(response) == 3.14
    
    def test_extract_answer_fallback_last_number(self, scorer):
        response = "I calculated 5 times 10 which gives 50"
        assert scorer.extract_answer(response) == 50.0
    
    def test_extract_answer_none(self, scorer):
        response = "I don't know the answer."
        assert scorer.extract_answer(response) is None
    
    def test_score_correct(self, scorer):
        response = "Let me solve this.\n2 + 2 = 4\n#### 4"
        reference = {"answer": "#### 4"}
        assert scorer.score("", response, reference) == 1.0
    
    def test_score_incorrect(self, scorer):
        response = "Let me solve this.\n2 + 2 = 5\n#### 5"
        reference = {"answer": "#### 4"}
        assert scorer.score("", response, reference) == 0.0
    
    def test_score_no_answer_extracted(self, scorer):
        response = "I don't know."
        reference = {"answer": "#### 4"}
        assert scorer.score("", response, reference) == 0.0
    
    def test_score_reference_without_hash(self, scorer):
        response = "The answer is 42."
        reference = {"answer": "42"}
        assert scorer.score("", response, reference) == 1.0
    
    def test_is_correct_true(self, scorer):
        response = "#### 100"
        reference = {"answer": "#### 100"}
        assert scorer.is_correct("", response, reference) is True
    
    def test_is_correct_false(self, scorer):
        response = "#### 99"
        reference = {"answer": "#### 100"}
        assert scorer.is_correct("", response, reference) is False

"""Tests for IFEval scorer."""

import pytest
from flipgate.scorers.ifeval import IFEvalScorer


@pytest.fixture
def scorer():
    return IFEvalScorer()


class TestIFEvalScorer:
    def test_score_no_instructions(self, scorer):
        response = "Any response"
        reference = {}
        assert scorer.score("", response, reference) == 1.0
    
    def test_score_keywords_present(self, scorer):
        response = "This response contains python and machine learning."
        reference = {
            "instruction_id_list": ["keywords"],
            "keywords": ["python", "machine learning"]
        }
        assert scorer.score("", response, reference) == 1.0
    
    def test_score_keywords_missing(self, scorer):
        response = "This response contains python."
        reference = {
            "instruction_id_list": ["keywords"],
            "keywords": ["python", "java"]
        }
        assert scorer.score("", response, reference) == 0.0
    
    def test_score_length_min_pass(self, scorer):
        response = "This is a response with enough words to pass the minimum check."
        reference = {
            "instruction_id_list": ["length"],
            "min_words": 5
        }
        assert scorer.score("", response, reference) == 1.0
    
    def test_score_length_min_fail(self, scorer):
        response = "Too short."
        reference = {
            "instruction_id_list": ["length"],
            "min_words": 10
        }
        assert scorer.score("", response, reference) == 0.0
    
    def test_score_length_max_fail(self, scorer):
        response = "This response has way too many words and exceeds the maximum limit."
        reference = {
            "instruction_id_list": ["length"],
            "max_words": 5
        }
        assert scorer.score("", response, reference) == 0.0
    
    def test_score_format_json_pass(self, scorer):
        response = '{"key": "value"}'
        reference = {
            "instruction_id_list": ["format"],
            "format": "json"
        }
        assert scorer.score("", response, reference) == 1.0
    
    def test_score_format_json_fail(self, scorer):
        response = "This is not JSON format."
        reference = {
            "instruction_id_list": ["format"],
            "format": "json"
        }
        assert scorer.score("", response, reference) == 0.0
    
    def test_score_format_bullet_pass(self, scorer):
        response = "- Item 1\n- Item 2\n- Item 3"
        reference = {
            "instruction_id_list": ["format"],
            "format": "bullet"
        }
        assert scorer.score("", response, reference) == 1.0
    
    def test_score_multiple_instructions_all_pass(self, scorer):
        response = "python programming with at least ten words in this response."
        reference = {
            "instruction_id_list": ["keywords", "length"],
            "keywords": ["python"],
            "min_words": 5
        }
        assert scorer.score("", response, reference) == 1.0
    
    def test_score_multiple_instructions_partial_pass(self, scorer):
        response = "python short."
        reference = {
            "instruction_id_list": ["keywords", "length"],
            "keywords": ["python"],
            "min_words": 10
        }
        assert scorer.score("", response, reference) == 0.5
    
    def test_is_correct_all_pass(self, scorer):
        response = "python programming with enough words."
        reference = {
            "instruction_id_list": ["keywords", "length"],
            "keywords": ["python"],
            "min_words": 3
        }
        assert scorer.is_correct("", response, reference) is True
    
    def test_is_correct_partial_fail(self, scorer):
        response = "python short."
        reference = {
            "instruction_id_list": ["keywords", "length"],
            "keywords": ["python"],
            "min_words": 10
        }
        assert scorer.is_correct("", response, reference) is False

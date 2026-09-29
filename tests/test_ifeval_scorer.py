"""Tests for IFEval scorer (real instruction format)."""

import pytest
from flipgate.scorers.ifeval import IFEvalScorer


@pytest.fixture
def scorer():
    return IFEvalScorer()


def ref(*pairs):
    """Build a reference dict from (instruction_id, kwargs) pairs."""
    return {
        "instruction_id_list": [p[0] for p in pairs],
        "kwargs": [p[1] for p in pairs],
    }


class TestPunctuation:
    def test_no_comma_pass(self, scorer):
        r = ref(("punctuation:no_comma", {}))
        assert scorer.is_correct("", "Hello world no commas here", r) is True

    def test_no_comma_fail(self, scorer):
        r = ref(("punctuation:no_comma", {}))
        assert scorer.is_correct("", "Hello, world", r) is False


class TestLengthConstraints:
    def test_number_words_at_least_pass(self, scorer):
        r = ref(("length_constraints:number_words", {"relation": "at least", "num_words": 5}))
        assert scorer.is_correct("", "one two three four five six", r) is True

    def test_number_words_at_least_fail(self, scorer):
        r = ref(("length_constraints:number_words", {"relation": "at least", "num_words": 5}))
        assert scorer.is_correct("", "one two three", r) is False

    def test_number_sentences_less_than(self, scorer):
        r = ref(("length_constraints:number_sentences", {"relation": "less than", "num_sentences": 3}))
        assert scorer.is_correct("", "First sentence. Second sentence.", r) is True
        assert scorer.is_correct("", "One. Two. Three. Four.", r) is False

    def test_number_paragraphs(self, scorer):
        r = ref(("length_constraints:number_paragraphs", {"relation": "at least", "num_paragraphs": 2}))
        assert scorer.is_correct("", "Para one.\n\nPara two.", r) is True
        assert scorer.is_correct("", "Only one paragraph.", r) is False

    def test_nth_paragraph_first_word_pass(self, scorer):
        r = ref(("length_constraints:nth_paragraph_first_word",
                 {"first_word": "weekend", "num_paragraphs": 2, "nth_paragraph": 1}))
        assert scorer.is_correct("", "weekend was great.\n\nSecond para.", r) is True

    def test_nth_paragraph_first_word_fail(self, scorer):
        r = ref(("length_constraints:nth_paragraph_first_word",
                 {"first_word": "weekend", "num_paragraphs": 2, "nth_paragraph": 2}))
        assert scorer.is_correct("", "First para.\n\nMonday was bad.", r) is False


class TestKeywords:
    def test_forbidden_words_pass(self, scorer):
        r = ref(("keywords:forbidden_words", {"forbidden_words": ["rock"]}))
        assert scorer.is_correct("", "I like jazz music", r) is True

    def test_forbidden_words_fail(self, scorer):
        r = ref(("keywords:forbidden_words", {"forbidden_words": ["rock"]}))
        assert scorer.is_correct("", "I like rock music", r) is False

    def test_existence_pass(self, scorer):
        r = ref(("keywords:existence", {"keywords": ["correlated", "experiencing"]}))
        assert scorer.is_correct("", "The correlated factors we are experiencing", r) is True

    def test_existence_fail(self, scorer):
        r = ref(("keywords:existence", {"keywords": ["correlated", "experiencing"]}))
        assert scorer.is_correct("", "The correlated factors", r) is False

    def test_frequency_pass(self, scorer):
        r = ref(("keywords:frequency", {"relation": "at least", "keyword": "story", "frequency": 2}))
        assert scorer.is_correct("", "A story about a story", r) is True

    def test_frequency_fail(self, scorer):
        r = ref(("keywords:frequency", {"relation": "at least", "keyword": "story", "frequency": 2}))
        assert scorer.is_correct("", "Just one story here", r) is False

    def test_letter_frequency_pass(self, scorer):
        r = ref(("keywords:letter_frequency",
                 {"let_relation": "at least", "letter": "#", "let_frequency": 4}))
        assert scorer.is_correct("", "#a #b #c #d", r) is True

    def test_letter_frequency_fail(self, scorer):
        r = ref(("keywords:letter_frequency",
                 {"let_relation": "at least", "letter": "#", "let_frequency": 4}))
        assert scorer.is_correct("", "#a #b", r) is False


class TestDetectableFormat:
    def test_highlights_pass(self, scorer):
        r = ref(("detectable_format:number_highlighted_sections", {"num_highlights": 2}))
        assert scorer.is_correct("", "See *part one* and *part two* here", r) is True

    def test_highlights_fail(self, scorer):
        r = ref(("detectable_format:number_highlighted_sections", {"num_highlights": 3}))
        assert scorer.is_correct("", "See *part one* only", r) is False

    def test_bullets_exact_pass(self, scorer):
        r = ref(("detectable_format:number_bullet_lists", {"num_bullets": 2}))
        assert scorer.is_correct("", "Intro\n* one\n* two", r) is True

    def test_bullets_exact_fail(self, scorer):
        r = ref(("detectable_format:number_bullet_lists", {"num_bullets": 3}))
        assert scorer.is_correct("", "Intro\n* one\n* two", r) is False

    def test_title_pass(self, scorer):
        r = ref(("detectable_format:title", {}))
        assert scorer.is_correct("", "Email body\n<<My Title>>\nMore text", r) is True

    def test_title_fail(self, scorer):
        r = ref(("detectable_format:title", {}))
        assert scorer.is_correct("", "Email body with no title", r) is False

    def test_json_format_pass(self, scorer):
        r = ref(("detectable_format:json_format", {}))
        assert scorer.is_correct("", '{"key": "value"}', r) is True

    def test_json_format_fenced_pass(self, scorer):
        r = ref(("detectable_format:json_format", {}))
        assert scorer.is_correct("", '```json\n{"key": "value"}\n```', r) is True

    def test_json_format_fail(self, scorer):
        r = ref(("detectable_format:json_format", {}))
        assert scorer.is_correct("", "This is not JSON.", r) is False

    def test_multiple_sections_pass(self, scorer):
        r = ref(("detectable_format:multiple_sections",
                 {"section_spliter": "PARAGRAPH", "num_sections": 2}))
        assert scorer.is_correct("", "PARAGRAPH 1 text\nPARAGRAPH 2 text", r) is True

    def test_multiple_sections_fail(self, scorer):
        r = ref(("detectable_format:multiple_sections",
                 {"section_spliter": "PARAGRAPH", "num_sections": 2}))
        assert scorer.is_correct("", "PARAGRAPH 1 only", r) is False

    def test_constrained_response_pass(self, scorer):
        prompt = "Choose from the following: ('My answer is yes.', 'My answer is no.')"
        r = ref(("detectable_format:constrained_response", {}))
        assert scorer.is_correct(prompt, "My answer is yes.", r) is True

    def test_constrained_response_fail(self, scorer):
        prompt = "Choose from the following: ('My answer is yes.', 'My answer is no.')"
        r = ref(("detectable_format:constrained_response", {}))
        assert scorer.is_correct(prompt, "I think maybe so.", r) is False


class TestDetectableContent:
    def test_placeholders_pass(self, scorer):
        r = ref(("detectable_content:number_placeholders", {"num_placeholders": 2}))
        assert scorer.is_correct("", "Name: [name], Address: [address]", r) is True

    def test_placeholders_fail(self, scorer):
        r = ref(("detectable_content:number_placeholders", {"num_placeholders": 3}))
        assert scorer.is_correct("", "Name: [name]", r) is False

    def test_postscript_pass(self, scorer):
        r = ref(("detectable_content:postscript", {"postscript_marker": "P.S."}))
        assert scorer.is_correct("", "Main letter.\nP.S. Don't forget.", r) is True

    def test_postscript_fail(self, scorer):
        r = ref(("detectable_content:postscript", {"postscript_marker": "P.S."}))
        assert scorer.is_correct("", "Main letter with no extra note.", r) is False


class TestLanguage:
    def test_kannada_pass(self, scorer):
        r = ref(("language:response_language", {"language": "kn"}))
        # Kannada script sample
        assert scorer.is_correct("", "ಹ್ಯಾಂಬರ್ಗರ್‌ಗಳು ಸ್ಯಾಂಡ್‌ವಿಚ್‌ಗಳಾಗಿವೆ ಹೌದು", r) is True

    def test_english_not_kannada(self, scorer):
        r = ref(("language:response_language", {"language": "kn"}))
        assert scorer.is_correct("", "Yes hamburgers are sandwiches", r) is False


class TestStartEnd:
    def test_quotation_pass(self, scorer):
        r = ref(("startend:quotation", {}))
        assert scorer.is_correct("", '"Quoted response here."', r) is True

    def test_quotation_fail(self, scorer):
        r = ref(("startend:quotation", {}))
        assert scorer.is_correct("", 'No quotes here.', r) is False

    def test_end_checker_pass(self, scorer):
        r = ref(("startend:end_checker", {"end_phrase": "Can I get my money back?"}))
        assert scorer.is_correct("", "Sorry. Can I get my money back?", r) is True

    def test_end_checker_fail(self, scorer):
        r = ref(("startend:end_checker", {"end_phrase": "Can I get my money back?"}))
        assert scorer.is_correct("", "Sorry, no refunds.", r) is False


class TestChangeCase:
    def test_lowercase_pass(self, scorer):
        r = ref(("change_case:english_lowercase", {}))
        assert scorer.is_correct("", "all lowercase text here", r) is True

    def test_lowercase_fail(self, scorer):
        r = ref(("change_case:english_lowercase", {}))
        assert scorer.is_correct("", "Has Uppercase Letter", r) is False

    def test_capital_pass(self, scorer):
        r = ref(("change_case:english_capital", {}))
        assert scorer.is_correct("", "ALL CAPS TEXT HERE", r) is True

    def test_capital_fail(self, scorer):
        r = ref(("change_case:english_capital", {}))
        assert scorer.is_correct("", "Has lowercase here", r) is False

    def test_capital_word_frequency(self, scorer):
        r = ref(("change_case:capital_word_frequency",
                 {"capital_relation": "less than", "capital_frequency": 3}))
        assert scorer.is_correct("", "Skills: PYTHON and SQL are great", r) is True
        assert scorer.is_correct("", "A B C D E F G too many", r) is False


class TestCombination:
    def test_repeat_prompt_pass(self, scorer):
        r = ref(("combination:repeat_prompt", {"prompt_to_repeat": "Write an email."}))
        assert scorer.is_correct("", "Write an email.\n\nDear boss...", r) is True

    def test_repeat_prompt_fail(self, scorer):
        r = ref(("combination:repeat_prompt", {"prompt_to_repeat": "Write an email."}))
        assert scorer.is_correct("", "Dear boss, I quit.", r) is False

    def test_two_responses_pass(self, scorer):
        r = ref(("combination:two_responses", {}))
        assert scorer.is_correct("", "First answer here. ****** Second answer here.", r) is True

    def test_two_responses_fail(self, scorer):
        r = ref(("combination:two_responses", {}))
        assert scorer.is_correct("", "Only one answer here.", r) is False


class TestScoring:
    def test_partial_credit(self, scorer):
        r = ref(
            ("punctuation:no_comma", {}),
            ("keywords:existence", {"keywords": ["python"]}),
        )
        # Passes no_comma, fails existence -> 0.5
        assert scorer.score("", "I like java programming", r) == 0.5

    def test_empty_instructions(self, scorer):
        assert scorer.score("", "anything", {}) == 1.0

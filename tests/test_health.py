"""Tests for harness-health checks and the v2 GSM8K scorer."""

from flipgate.health import RunHealth, compare_health, finish_reason, run_health
from flipgate.scorers.gsm8k import GSM8KScorer, GSM8KScorerV1


def item(response="done", reason=None, tokens=None):
    md = {}
    if reason:
        md["finish_reason"] = reason
    if tokens is not None:
        md["n_new_tokens"] = tokens
    return {"item_id": "x", "response": response, "score": 1.0, "metadata": md} if md else {"item_id": "x", "response": response, "score": 1.0}


class TestFinishReason:
    def test_eos_is_stop_even_at_cap(self):
        assert finish_reason(256, 256, ended_with_eos=True) == "stop"

    def test_cap_without_eos_is_length(self):
        assert finish_reason(256, 256, ended_with_eos=False) == "length"

    def test_short_without_eos_is_stop(self):
        assert finish_reason(40, 256, ended_with_eos=False) == "stop"


class TestRunHealth:
    def test_truncation_rate_from_finish_reason(self):
        items = [item(reason="length")] * 3 + [item(reason="stop")] * 7
        h = run_health(items)
        assert h.truncated_rate == 0.3
        assert h.n_with_finish_info == 10

    def test_unknown_when_no_metadata(self):
        h = run_health([item(), item()])
        assert h.truncated_rate is None

    def test_gsm8k_no_final_answer_rate(self):
        items = [item("work work \\boxed{5}"), item("cut off mid-sentence 2 +"), item("so the answer is 7")]
        h = run_health(items, dataset="gsm8k")
        assert abs(h.no_final_answer_rate - 1 / 3) < 1e-9


class TestCompareHealth:
    def h(self, rate):
        return RunHealth(n=100, n_with_finish_info=100, truncated_rate=rate, no_final_answer_rate=None, median_chars=500)

    def test_clean_runs_pass(self):
        problems, warnings = compare_health(self.h(0.01), self.h(0.02))
        assert problems == [] and warnings == []

    def test_heavy_truncation_is_a_problem(self):
        problems, _ = compare_health(self.h(0.84), self.h(0.92))
        assert any("hit the generation cap" in p for p in problems)

    def test_rate_gap_is_a_problem_even_below_the_limit(self):
        problems, _ = compare_health(self.h(0.00), self.h(0.08), max_truncation=0.10, max_gap=0.05)
        assert any("differ" in p for p in problems)

    def test_unknown_finish_info_warns_not_fails(self):
        unknown = RunHealth(100, 0, None, 0.9, 800)
        problems, warnings = compare_health(unknown, unknown)
        assert problems == []
        assert any("cannot be measured" in w for w in warnings)
        assert any("no final-answer marker" in w for w in warnings)


class TestGSM8KScorerV2:
    ref = {"answer": "work\n#### 1,250"}

    def test_boxed_answer_with_thousands_separator(self):
        assert GSM8KScorer().score("q", "so \\(\\boxed{1,250}\\)", self.ref) == 1.0

    def test_v1_misreads_thousands_separator(self):
        # documents the original weakness: "1,250" is read as 250
        assert GSM8KScorerV1().extract_answer("the answer is 1,250") == 1.0

    def test_last_boxed_wins(self):
        assert GSM8KScorer().extract_answer("\\boxed{3} then corrected \\boxed{4}") == 4.0

    def test_truncated_response_scores_zero(self):
        assert GSM8KScorer().score("q", "so the total is 80,000 + 120,0", self.ref) == 0.0

    def test_hash_marker_still_works(self):
        assert GSM8KScorer().extract_answer("blah\n#### 42") == 42.0

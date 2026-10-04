"""`flipgate check` must refuse to certify a comparison whose responses were truncated."""

from click.testing import CliRunner

from flipgate.cli import main
from flipgate.store import ResultsStore


def make_run(store, run_id, scores, reason):
    for i, score in enumerate(scores):
        store.append_item(run_id, "gsm8k", f"item_{i:04d}", "q", "a", score,
                          metadata={"finish_reason": reason(i), "n_new_tokens": 10})


def run_check(tmp_path, base_reason, cand_reason, extra=()):
    store = ResultsStore(tmp_path)
    scores_b = [1.0] * 60 + [0.0] * 40
    scores_c = [1.0] * 58 + [0.0] * 2 + [0.0] * 40   # nearly identical: no significant flips
    make_run(store, "base", scores_b, base_reason)
    make_run(store, "cand", scores_c, cand_reason)
    return CliRunner().invoke(main, ["check", "-b", "base", "-c", "cand", "-d", "gsm8k", "-r", str(tmp_path), *extra])


def test_clean_runs_are_not_invalid(tmp_path):
    result = run_check(tmp_path, lambda i: "stop", lambda i: "stop")
    assert result.exit_code in (0, 1)
    assert "INVALID" not in result.output


def test_heavily_truncated_runs_are_invalid(tmp_path):
    result = run_check(tmp_path, lambda i: "length" if i % 10 else "stop", lambda i: "length")
    assert result.exit_code == 2
    assert "INVALID" in " ".join(result.output.split())


def test_allow_truncation_overrides(tmp_path):
    result = run_check(tmp_path, lambda i: "length", lambda i: "length", extra=("--allow-truncation",))
    assert result.exit_code != 2
    assert "HEALTH" in " ".join(result.output.split())


def test_unknown_finish_info_only_warns(tmp_path):
    store = ResultsStore(tmp_path)
    for run in ("base", "cand"):
        for i in range(30):
            store.append_item(run, "gsm8k", f"item_{i:04d}", "q", "a", 1.0)
    result = CliRunner().invoke(main, ["check", "-b", "base", "-c", "cand", "-d", "gsm8k", "-r", str(tmp_path)])
    assert result.exit_code == 0
    assert "cannot be measured" in " ".join(result.output.split())

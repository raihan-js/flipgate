"""Harness health: detect generation truncation before trusting a flip analysis.

A flip analysis compares per-item pass/fail between two runs. If either run cut its
responses off at the generation cap, "wrong" mostly means "did not finish", and a candidate
that is more (or less) verbose than the baseline will look better (or worse) for reasons
unrelated to quality. FlipGate's own GSM8K sweeps hit exactly this (a 256-token cap), so
`flipgate check` now measures it and refuses to certify a comparison it cannot trust.

Items may carry ``metadata = {"finish_reason": "stop" | "length", "n_new_tokens": int}``.
Without it the truncation rate is unknown; for GSM8K a "no final answer marker" rate is
reported as a weaker signal.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any, Iterable

# Markers Qwen-style math answers use when they actually finish.
_GSM8K_FINAL = re.compile(r"\\boxed\{|####|answer is", re.I)


@dataclass
class RunHealth:
    n: int
    n_with_finish_info: int
    truncated_rate: float | None  # share that hit the generation cap; None if unknown
    no_final_answer_rate: float | None  # gsm8k-only heuristic; None otherwise
    median_chars: int


def finish_reason(n_new_tokens: int, max_new_tokens: int, ended_with_eos: bool) -> str:
    """'stop' if the model ended on its own, 'length' if it ran into the cap."""
    if ended_with_eos:
        return "stop"
    return "length" if n_new_tokens >= max_new_tokens else "stop"


def run_health(items: Iterable[dict[str, Any]], dataset: str | None = None) -> RunHealth:
    items = list(items)
    n = len(items)
    reasons = [(it.get("metadata") or {}).get("finish_reason") for it in items]
    known = [r for r in reasons if r in ("stop", "length")]
    truncated = (sum(1 for r in known if r == "length") / len(known)) if known else None
    no_final = None
    if dataset == "gsm8k" and n:
        no_final = sum(1 for it in items if not _GSM8K_FINAL.search(it.get("response", ""))) / n
    lengths = [len(it.get("response", "")) for it in items] or [0]
    return RunHealth(n, len(known), truncated, no_final, int(statistics.median(lengths)))


def compare_health(
    baseline: RunHealth,
    candidate: RunHealth,
    max_truncation: float = 0.10,
    max_gap: float = 0.05,
) -> tuple[list[str], list[str]]:
    """Return (problems, warnings). Any problem makes the comparison INVALID."""
    problems: list[str] = []
    warnings: list[str] = []
    for name, h in (("baseline", baseline), ("candidate", candidate)):
        if h.truncated_rate is None:
            warnings.append(f"{name}: no finish_reason recorded, so truncation cannot be measured")
            if h.no_final_answer_rate is not None and h.no_final_answer_rate > 0.5:
                warnings.append(
                    f"{name}: {h.no_final_answer_rate:.0%} of responses contain no final-answer marker; "
                    "check the generation cap"
                )
        elif h.truncated_rate > max_truncation:
            problems.append(
                f"{name}: {h.truncated_rate:.0%} of responses hit the generation cap (limit {max_truncation:.0%})"
            )
    if baseline.truncated_rate is not None and candidate.truncated_rate is not None:
        gap = abs(baseline.truncated_rate - candidate.truncated_rate)
        if gap > max_gap:
            problems.append(
                f"truncation rates differ by {gap:.0%} between baseline and candidate; "
                "verbosity, not quality, may explain the flips"
            )
    return problems, warnings

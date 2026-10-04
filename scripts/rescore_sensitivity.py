"""Scorer-sensitivity check for the 1,000-item GSM8K sweep.

Re-scores the stored responses of the three GSM8K runs with a more robust answer
extractor, and reports completion rate, accuracy and flips next to the repo scorer.
Reads the published dataset, so anyone can reproduce the numbers in the README:

    python scripts/rescore_sensitivity.py --parquet data/train-00000-of-00001.parquet
    # parquet: https://huggingface.co/datasets/raihan-js/flipgate-results

Needs pyarrow and datasets (GSM8K test split is only used for the reference answers).
"""
import argparse
import math
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flipgate.scorers.gsm8k import GSM8KScorer  # noqa: E402

RUNS = {  # run ids as published in the dataset
    "bf16": "Qwen2.5-3B-bf16_hf_generate_gsm8k_bs1_814ab601",
    "AWQ": "Qwen2.5-3B-AWQ_hf_generate_gsm8k_bs1_22e83eed",
    "GPTQ-Int4": "Qwen2.5-3B-GPTQ-Int4_hf_generate_gsm8k_bs1_afa1d64f",
}
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def to_num(s):
    try:
        return float(s.replace(",", "").replace("$", "").strip().rstrip("."))
    except ValueError:
        return None


def robust_extract(resp):
    """Last \\boxed{}, then '####', then 'answer is', then the last number (thousands separators handled)."""
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", resp)
    if boxed and (m := NUM.findall(boxed[-1].replace("\\,", ""))):
        return to_num(m[-1])
    if m := re.search(r"####\s*(-?\d[\d,]*\.?\d*)", resp):
        return to_num(m.group(1))
    if m := re.search(r"answer is[^\d\-]{0,20}(-?\d[\d,]*\.?\d*)", resp, re.I):
        return to_num(m.group(1))
    nums = NUM.findall(resp)
    return to_num(nums[-1]) if nums else None


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    args = ap.parse_args()
    table = {r["run_id"]: r["results"]["gsm8k"] for r in pq.read_table(args.parquet).to_pylist() if r["results"].get("gsm8k")}
    ref = [float(r["answer"].split("####")[-1].strip().replace(",", "")) for r in load_dataset("openai/gsm8k", "main", split="test")]
    repo_scorer, out = GSM8KScorer(), {}
    for name, run_id in RUNS.items():
        rows = table[run_id]
        stored = {r["item_id"]: r["score"] for r in rows}
        robust, boxed = {}, 0
        for r in rows:
            gold = ref[int(re.search(r"\d+", r["item_id"]).group())]
            p = robust_extract(r["response"])
            robust[r["item_id"]] = 1.0 if p is not None and abs(p - gold) < 1e-6 else 0.0
            boxed += "boxed" in r["response"]
        done_acc = sum(robust[r["item_id"]] for r in rows if "boxed" in r["response"]) / max(1, boxed)
        out[name] = (stored, robust)
        n = len(rows)
        print(f"{name:10} n={n}  repo scorer {100*sum(stored.values())/n:.1f}%  robust {100*sum(robust.values())/n:.1f}%  "
              f"boxed final answer {100*boxed/n:.1f}%  accuracy when boxed {100*done_acc:.1f}%")
    for label, ix in (("repo scorer (stored)", 0), ("robust extractor", 1)):
        print(f"\nflips vs bf16, {label}")
        base = out["bf16"][ix]
        for name in ("AWQ", "GPTQ-Int4"):
            cand = out[name][ix]
            rw = sum(1 for i in base if base[i] == 1 and cand[i] == 0)
            wr = sum(1 for i in base if base[i] == 0 and cand[i] == 1)
            print(f"  {name:10} right->wrong {rw}  wrong->right {wr}  net {(wr - rw) / 10:+.1f} pts  McNemar p = {mcnemar_exact(rw, wr):.3g}")


if __name__ == "__main__":
    main()

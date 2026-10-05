#!/usr/bin/env python3
"""Export every FlipGate run to flat Parquet tables for the Hugging Face dataset (rewritten 2026-10-05).

  items.parquet : one row per (run, item): prompt, response, score, finish_reason, n_new_tokens, score_v1
  runs.parquet  : one row per run: model, engine, dataset, batch size, accuracy, cap, truncation rate, status

  python scripts/publish_hf.py --out <dir>           write the tables
  python scripts/publish_hf.py --out <dir> --upload  also upload them (token from ~/.cache/huggingface/token)
"""
import argparse
import ast
import json
from pathlib import Path


def parse_meta(v):
    if isinstance(v, dict):
        return v
    if not v:
        return {}
    try:
        return json.loads(v)
    except Exception:
        try:
            return ast.literal_eval(v)
        except Exception:
            return {}


def status(m):
    ds, eng = m.get("dataset"), m.get("engine")
    if ds != "gsm8k":
        return "current"
    if "max_new_tokens" in m:
        return "current: 1,024-token cap, finish reasons recorded"
    if eng == "llama_cpp":
        return "withdrawn: 256-token cap and a different prompt from the HF runs"
    if m.get("num_items") == 1000:
        return "superseded: 256-token cap, most answers truncated"
    return "early small run: generation cap not recorded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    import pandas as pd

    runs, items = [], []
    for d in sorted(Path(args.results).iterdir()):
        mf = d / "metadata.json"
        if not d.is_dir() or not mf.exists():
            continue
        m = json.load(open(mf))
        n = 0
        for jf in sorted(d.glob("*.jsonl")):
            for line in open(jf):
                if not line.strip():
                    continue
                r = json.loads(line)
                if "item_id" not in r:
                    continue
                meta = parse_meta(r.get("metadata"))
                try:
                    score = float(r.get("score"))
                except (TypeError, ValueError):
                    score = None
                items.append({"run_id": d.name, "model": m.get("model"), "engine": m.get("engine"),
                              "dataset": m.get("dataset"), "batch_size": m.get("batch_size"),
                              "item_id": r["item_id"], "prompt": r.get("prompt"), "response": r.get("response"),
                              "score": score, "finish_reason": meta.get("finish_reason"),
                              "n_new_tokens": meta.get("n_new_tokens"), "score_v1": meta.get("score_v1"),
                              "timestamp": r.get("timestamp")})
                n += 1
        runs.append({"run_id": d.name, "model": m.get("model"), "engine": m.get("engine"),
                     "engine_version": m.get("engine_version"), "dataset": m.get("dataset"),
                     "batch_size": m.get("batch_size"), "num_items": n, "accuracy": m.get("accuracy"),
                     "accuracy_v1_scorer": m.get("accuracy_v1_scorer"), "max_new_tokens": m.get("max_new_tokens"),
                     "truncated_rate": m.get("truncated_rate"), "manifest_version": m.get("manifest_sha"),
                     "created": m.get("created"), "status": status(m)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(items).to_parquet(out / "items.parquet", index=False)
    pd.DataFrame(runs).to_parquet(out / "runs.parquet", index=False)
    print(f"{len(runs)} runs, {len(items)} items -> {out}")
    if args.upload:
        from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
        card = (out / "README.md")
        ops = [CommitOperationAdd("items.parquet", str(out / "items.parquet")),
               CommitOperationAdd("runs.parquet", str(out / "runs.parquet")),
               CommitOperationAdd("README.md", str(card)),
               CommitOperationDelete("data/train-00000-of-00001.parquet")]
        HfApi().create_commit("raihan-js/flipgate-results", ops, repo_type="dataset",
                              commit_message="Re-export as flat tables (items, runs); add the 1,024-token GSM8K re-run and the batch-size noise floor; flag superseded runs")
        print("uploaded")


if __name__ == "__main__":
    main()

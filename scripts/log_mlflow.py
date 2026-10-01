#!/usr/bin/env python3
"""Log all stored FlipGate runs to MLflow (post-hoc).

Reads every run's metadata + scores from the JSONL store and logs
accuracy, item count, engine, and timing to a local MLflow experiment.
Usage: PYTHONPATH=src python scripts/log_mlflow.py
"""
from flipgate.mlflow_logger import log_run_to_mlflow
from flipgate.store import ResultsStore

DATASETS = ["gsm8k", "ifeval", "fedproc", "bfcl_simple"]


def main() -> None:
    store = ResultsStore("data/results")
    n = 0
    for run_id in sorted(store.list_runs()):
        try:
            meta = store.load_run_metadata(run_id)
        except FileNotFoundError:
            continue
        ds = meta.get("dataset", "gsm8k")
        items = list(store.iter_items(run_id, ds))
        if not items:
            continue
        correct = sum(1 for it in items if it["score"] == 1.0)
        extra = dict(meta.get("extra", {}) or {})
        log_run_to_mlflow(
            run_id=run_id,
            model=meta.get("model", "unknown"),
            engine=meta.get("engine", "unknown"),
            dataset=ds,
            accuracy=correct / len(items),
            flip_rate=0.0,  # pairwise flip rates live in gate reports, not per-run
            metadata={"num_items": len(items), "batch_size": meta.get("batch_size"),
                      "elapsed": extra.get("elapsed")},
        )
        n += 1
    print(f"Logged {n} runs to ./mlruns (experiment 'flipgate')")


if __name__ == "__main__":
    main()

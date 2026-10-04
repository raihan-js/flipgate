#!/usr/bin/env python3
"""ORIGINAL GSM8K runner, kept for provenance. DO NOT USE for new runs.

This script hard-codes max_new_tokens=256 and generates one item at a time. The first GSM8K sweeps
(33.0% / 43.1% / 37.3%) came from it, and most of their answers were truncated by that cap.
Use scripts/run_gsm8k.py, which takes the cap from configs/manifest.yaml and records finish reasons.

GSM8K evaluation runner (resumable).

Uses item_XXXX IDs over the test split prefix, so runs extend naturally:
first run 200 items, resume with --run-id to reach 1000.
Usage: PYTHONPATH=src python scripts/run_gsm8k.py --model <path> --model-name <name> [--run-id <id>] [--n-items 1000]
"""
import argparse
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from flipgate.scorers.gsm8k import GSM8KScorer
from flipgate.store import ResultsStore, generate_run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--n-items", type=int, default=1000)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    print("Loading GSM8K test set...", flush=True)
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    items = [{"question": it["question"], "answer": it["answer"]} for it in gsm8k][:args.n_items]
    print(f"Using {len(items)} items", flush=True)

    store = ResultsStore("data/results")
    run_id = args.run_id or generate_run_id(args.model_name, "hf_generate", "gsm8k", 1)
    done = set()
    if args.run_id:
        done = {it["item_id"] for it in store.iter_items(run_id, "gsm8k")}
        print(f"Resuming {run_id}: {len(done)} done", flush=True)
    else:
        print(f"Run ID: {run_id}", flush=True)

    todo = [(i, it) for i, it in enumerate(items) if f"item_{i:04d}" not in done]
    print(f"Remaining: {len(todo)}", flush=True)
    if not todo:
        print("Nothing to do.", flush=True)
        return

    print(f"Loading {args.model_name}...", flush=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="cuda",
        pad_token_id=tokenizer.pad_token_id)
    print("Model loaded", flush=True)

    scorer = GSM8KScorer()
    correct = sum(1 for it in store.iter_items(run_id, "gsm8k") if it["score"] == 1.0)
    start = time.time()
    for n, (i, item) in enumerate(todo, 1):
        messages = [{"role": "user", "content": item["question"]}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True)
        score = scorer.score(item["question"], response, {"answer": item["answer"]})
        if score == 1.0:
            correct += 1
        store.append_item(run_id, "gsm8k", f"item_{i:04d}",
                          item["question"], response, score)
        if n % 20 == 0:
            done_total = len(done) + n
            print(f"Progress: {done_total}/{len(items)}, acc: {correct/done_total:.1%}",
                  flush=True)

    elapsed = time.time() - start
    total = len(items)
    accuracy = correct / total
    store.append_run_metadata(run_id, args.model_name, "hf_generate", "5.17.0",
                              "gsm8k", "test", batch_size=1,
                              extra={"accuracy": accuracy, "num_items": total,
                                     "elapsed": elapsed})
    print(f"Done! {accuracy:.1%} ({correct}/{total}) in {elapsed:.1f}s", flush=True)
    print(f"Run ID: {run_id}", flush=True)


if __name__ == "__main__":
    main()

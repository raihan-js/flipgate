#!/usr/bin/env python3
"""GSM8K evaluation runner (resumable, batched, truncation-aware).

Differences from the original runner (kept as run_gsm8k_v1_cap256.py):
  * the generation cap comes from configs/manifest.yaml (sampling.max_tokens), not a hard-coded 256
  * left-padded batching (greedy; batch determinism can be checked with --determinism-check)
  * every item records finish_reason ("stop" | "length") and n_new_tokens, so `flipgate check`
    can refuse a comparison whose responses were cut off
  * stores the robust v2 score as `score` and the original v1 score in metadata, so both can be reported

Usage:
  PYTHONPATH=src python scripts/run_gsm8k.py --model Qwen/Qwen2.5-3B-Instruct --model-name Qwen2.5-3B-bf16 \\
      [--run-id <id>] [--n-items 1000] [--batch-size 8] [--max-new-tokens N] [--determinism-check 40]
"""
import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from flipgate.health import finish_reason, run_health
from flipgate.manifest import load_manifest
from flipgate.scorers.gsm8k import GSM8KScorer, GSM8KScorerV1
from flipgate.store import ResultsStore, generate_run_id


def eos_ids(model):
    e = model.generation_config.eos_token_id
    return set(e) if isinstance(e, (list, tuple)) else ({e} if e is not None else set())


def generate_batch(model, tokenizer, prompts, max_new_tokens):
    """Greedy generation for a list of formatted prompts. Returns [(text, n_new_tokens, finish_reason)]."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    stops, results = eos_ids(model), []
    for row in out:
        gen = row[inputs.input_ids.shape[1]:].tolist()
        first_eos = next((k for k, t in enumerate(gen) if t in stops), None)
        n_new = (first_eos + 1) if first_eos is not None else len(gen)
        text = tokenizer.decode(gen[:n_new], skip_special_tokens=True)
        results.append((text, n_new, finish_reason(n_new, max_new_tokens, first_eos is not None)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--n-items", type=int, default=1000)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default: manifest sampling.max_tokens")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--determinism-check", type=int, default=0,
                    help="generate N items at batch size 1 and at --batch-size, report mismatches, then exit")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    cap = args.max_new_tokens or manifest["sampling"]["max_tokens"]
    print(f"manifest {manifest['manifest_version']}, generation cap {cap} tokens, batch size {args.batch_size}", flush=True)

    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    items = [{"question": it["question"], "answer": it["answer"]} for it in gsm8k][:args.n_items]

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=args.device,
                                                 pad_token_id=tokenizer.pad_token_id)
    fmt = lambda q: tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                                  add_generation_prompt=True)

    if args.determinism_check:
        sample = [fmt(it["question"]) for it in items[:args.determinism_check]]
        single = [generate_batch(model, tokenizer, [p], cap)[0][0] for p in sample]
        batched = []
        for i in range(0, len(sample), args.batch_size):
            batched += [r[0] for r in generate_batch(model, tokenizer, sample[i:i + args.batch_size], cap)]
        mism = sum(1 for a, b in zip(single, batched) if a != b)
        print(f"determinism check: {mism}/{len(sample)} responses differ between batch size 1 and {args.batch_size}")
        return

    store = ResultsStore(args.results_dir)
    run_id = args.run_id or generate_run_id(args.model_name, "hf_generate", "gsm8k", args.batch_size)
    done = {it["item_id"] for it in store.iter_items(run_id, "gsm8k")} if args.run_id else set()
    todo = [(i, it) for i, it in enumerate(items) if f"item_{i:04d}" not in done]
    print(f"run {run_id}: {len(done)} done, {len(todo)} to do", flush=True)

    v2, v1 = GSM8KScorer(), GSM8KScorerV1()
    start = time.time()
    for b in range(0, len(todo), args.batch_size):
        chunk = todo[b:b + args.batch_size]
        outs = generate_batch(model, tokenizer, [fmt(it["question"]) for _, it in chunk], cap)
        for (i, item), (text, n_new, reason) in zip(chunk, outs):
            ref = {"answer": item["answer"]}
            store.append_item(run_id, "gsm8k", f"item_{i:04d}", item["question"], text,
                              v2.score(item["question"], text, ref),
                              metadata={"finish_reason": reason, "n_new_tokens": n_new,
                                        "score_v1": v1.score(item["question"], text, ref)})
        if (b // args.batch_size) % 5 == 0:
            print(f"progress {len(done) + b + len(chunk)}/{len(items)}", flush=True)

    rows = list(store.iter_items(run_id, "gsm8k"))
    health = run_health(rows, "gsm8k")
    acc = sum(1 for r in rows if r["score"] == 1.0) / len(rows)
    acc_v1 = sum(1 for r in rows if (r.get("metadata") or {}).get("score_v1") == 1.0) / len(rows)
    store.append_run_metadata(run_id, args.model_name, "hf_generate", "5.17.0", "gsm8k",
                              manifest["manifest_version"], batch_size=args.batch_size,
                              extra={"accuracy": acc, "accuracy_v1_scorer": acc_v1, "num_items": len(rows),
                                     "max_new_tokens": cap, "scorer_version": GSM8KScorer.version,
                                     "truncated_rate": health.truncated_rate,
                                     "elapsed": time.time() - start})
    print(f"done: accuracy {acc:.1%} (v1 scorer {acc_v1:.1%}), truncated {health.truncated_rate:.1%}, run {run_id}")
    if health.truncated_rate and health.truncated_rate > 0.10:
        print("WARNING: more than 10% of responses hit the cap; raise --max-new-tokens before trusting flips")


if __name__ == "__main__":
    main()

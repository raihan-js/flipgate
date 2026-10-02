#!/usr/bin/env python3
"""vLLM noise floor: bf16-vs-bf16 flips at fixed batch size, temp 0.

Same model, same dtype, same greedy sampling params, run twice. Any
right->wrong flip is pure serving nondeterminism (batched kernel
reduction order), i.e. the floor a quantization regression must clear.

Batch composition is the variable: prompts are chunked into groups of
`batch_size` and each chunk is generated as one scheduler batch, which
is how a real serving stack drives throughput.

Usage:
  PYTHONPATH=src python scripts/vllm_noise_floor.py --batch-sizes 32,8,1 --passes 2
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("data/noise_floor")
MODEL = "data/models/Qwen--Qwen2.5-3B-Instruct"


def load_items(n_items: int) -> list[dict]:
    """First n_items of GSM8K test - the frozen FlipGate slice."""
    from datasets import load_dataset
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    return [{"item_id": f"item_{i:04d}", "question": it["question"], "answer": it["answer"]}
            for i, it in enumerate(gsm8k)][:n_items]


def pass_file(batch_size: int, pass_idx: int) -> Path:
    return OUT_DIR / f"bs{batch_size}_pass{pass_idx}.json"


def load_pass(batch_size: int, pass_idx: int) -> dict:
    p = pass_file(batch_size, pass_idx)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save_pass(batch_size: int, pass_idx: int, data: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pass_file(batch_size, pass_idx).write_text(json.dumps(data))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", default="32,8,1")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--n-items", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    items = load_items(args.n_items)
    print(f"Items: {len(items)} | batch sizes: {batch_sizes} | passes: {args.passes}",
          flush=True)

    from vllm import LLM, SamplingParams
    from flipgate.scorers.gsm8k import GSM8KScorer

    scorer = GSM8KScorer()
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.85, enforce_eager=True,
              disable_log_stats=True)
    llm.chat_template = getattr(llm, "chat_template", None)

    throughput = {}
    for batch_size in batch_sizes:
        for pass_idx in range(args.passes):
            done = load_pass(batch_size, pass_idx)
            todo = [it for it in items if it["item_id"] not in done]
            if not todo:
                print(f"[bs={batch_size} pass={pass_idx}] complete", flush=True)
                continue
            print(f"[bs={batch_size} pass={pass_idx}] {len(todo)} items to go", flush=True)

            t0 = time.time()
            gen_tokens = 0
            for start in range(0, len(todo), batch_size):
                chunk = todo[start:start + batch_size]
                prompts = [f"<|im_start|>user\n{c['question']}<|im_end|>\n"
                           f"<|im_start|>assistant\n" for c in chunk]
                outs = llm.generate(prompts, sampling)
                for c, o in zip(chunk, outs):
                    text = o.outputs[0].text
                    gen_tokens += len(o.outputs[0].token_ids)
                    done[c["item_id"]] = {
                        "score": float(scorer.score(c["question"], text,
                                                    {"answer": c["answer"]})),
                        "finish": o.outputs[0].finish_reason,
                        "n_tokens": len(o.outputs[0].token_ids),
                    }
                if (start // batch_size) % 5 == 0:
                    save_pass(batch_size, pass_idx, done)
                    print(f"  {start + len(chunk)}/{len(todo)} "
                          f"({gen_tokens / max(time.time() - t0, 1e-9):.1f} tok/s)",
                          flush=True)
            elapsed = time.time() - t0
            save_pass(batch_size, pass_idx, done)
            throughput[f"bs{batch_size}_pass{pass_idx}"] = {
                "items": len(done), "seconds": round(elapsed, 1),
                "gen_tokens": gen_tokens,
                "tokens_per_s": round(gen_tokens / elapsed, 1),
            }
            acc = np.mean([v["score"] for v in done.values()])
            print(f"[bs={batch_size} pass={pass_idx}] acc={acc:.3f} "
                  f"{elapsed:.0f}s {gen_tokens / elapsed:.1f} tok/s", flush=True)

    # ---- analysis: pass 0 vs pass 1 at each batch size ----
    from flipgate.stats.bootstrap import paired_bootstrap_ci
    from flipgate.stats.mcnemar import count_flips, mcnemar_test

    report = {"model": MODEL, "engine": "vllm", "n_items": args.n_items,
              "temperature": 0.0, "max_tokens": args.max_tokens,
              "throughput": throughput, "by_batch_size": {}}

    for batch_size in batch_sizes:
        a, b = load_pass(batch_size, 0), load_pass(batch_size, 1)
        ids = sorted(set(a) & set(b))
        base = np.array([a[i]["score"] == 1.0 for i in ids])
        cand = np.array([b[i]["score"] == 1.0 for i in ids])
        flips = count_flips(base, cand)
        boot = paired_bootstrap_ci(base, cand, direction="right_to_wrong")
        mac = mcnemar_test(base, cand)
        text_flips = 0
        report["by_batch_size"][str(batch_size)] = {
            "n": len(ids),
            "acc_pass0": round(float(base.mean()), 4),
            "acc_pass1": round(float(cand.mean()), 4),
            **flips,
            "r2w_rate": round(flips["right_to_wrong"] / len(ids), 4),
            "r2w_ci95": [round(boot.ci_lower, 4), round(boot.ci_upper, 4)],
            "mcnemar_p": round(float(mac.p_value), 4),
            "identical_scores": flips["total_flips"] == 0,
        }
        print(f"\n=== batch size {batch_size} (n={len(ids)}) ===")
        print(f"  acc {base.mean():.3f} vs {cand.mean():.3f}")
        print(f"  right->wrong {flips['right_to_wrong']}, wrong->right {flips['wrong_to_right']}")
        print(f"  r2w rate {flips['right_to_wrong'] / len(ids):.4f} "
              f"95% CI [{boot.ci_lower:.4f}, {boot.ci_upper:.4f}]")
        print(f"  McNemar p={mac.p_value:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
"""GSM8K on vLLM (eager, temp 0) with the same prompt, cap and scorer as scripts/run_gsm8k.py.

Gives (a) an engine-only comparison: the same bf16 weights on vLLM vs HF generate, and (b) a vLLM
noise floor at the 1,024-token cap (repeat passes at one batch size, and different batch sizes).
Each pass is stored as a normal FlipGate run, so `flipgate check` works on it.

  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTHONPATH=src \\
    python scripts/run_gsm8k_vllm.py --run-id Qwen2.5-3B-bf16-vllm_gsm8k_v2_bs32_pass0 --batch-size 32
"""
import argparse
import time

from datasets import load_dataset
from transformers import AutoTokenizer

from flipgate.health import run_health
from flipgate.manifest import load_manifest
from flipgate.scorers.gsm8k import GSM8KScorer, GSM8KScorerV1
from flipgate.store import ResultsStore

MODEL = "data/models/Qwen--Qwen2.5-3B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--model-name", default="Qwen2.5-3B-bf16-vllm")
    ap.add_argument("--batch-size", type=int, default=32, help="prompts handed to the scheduler at once")
    ap.add_argument("--n-items", type=int, default=1000)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--manifest", default="configs/manifest.yaml")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    cap = args.max_new_tokens or manifest["sampling"]["max_tokens"]
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    items = [{"question": it["question"], "answer": it["answer"]} for it in gsm8k][:args.n_items]
    tok = AutoTokenizer.from_pretrained(args.model)
    fmt = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)

    store = ResultsStore("data/results")
    done = {it["item_id"] for it in store.iter_items(args.run_id, "gsm8k")}
    todo = [(i, it) for i, it in enumerate(items) if f"item_{i:04d}" not in done]
    print(f"run {args.run_id}: {len(done)} done, {len(todo)} to do, cap {cap}, batch {args.batch_size}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True, disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=cap)
    v2, v1 = GSM8KScorer(), GSM8KScorerV1()
    start = time.time()
    for b in range(0, len(todo), args.batch_size):
        chunk = todo[b:b + args.batch_size]
        outs = llm.generate([fmt(it["question"]) for _, it in chunk], sampling, use_tqdm=False)
        for (i, item), out in zip(chunk, outs):
            o = out.outputs[0]
            text, reason = o.text, ("length" if o.finish_reason == "length" else "stop")
            ref = {"answer": item["answer"]}
            store.append_item(args.run_id, "gsm8k", f"item_{i:04d}", item["question"], text,
                              v2.score(item["question"], text, ref),
                              metadata={"finish_reason": reason, "n_new_tokens": len(o.token_ids),
                                        "score_v1": v1.score(item["question"], text, ref)})
        if (b // args.batch_size) % 5 == 0:
            print(f"progress {len(done) + b + len(chunk)}/{len(items)}", flush=True)

    rows = list(store.iter_items(args.run_id, "gsm8k"))
    health = run_health(rows, "gsm8k")
    acc = sum(1 for r in rows if r["score"] == 1.0) / len(rows)
    store.append_run_metadata(args.run_id, args.model_name, "vllm", "0.30.0", "gsm8k", manifest["manifest_version"],
                              batch_size=args.batch_size,
                              extra={"accuracy": acc, "num_items": len(rows), "max_new_tokens": cap,
                                     "scorer_version": GSM8KScorer.version, "truncated_rate": health.truncated_rate,
                                     "elapsed": time.time() - start})
    print(f"done: accuracy {acc:.1%}, truncated {health.truncated_rate:.1%}, run {args.run_id}", flush=True)


if __name__ == "__main__":
    main()

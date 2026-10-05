"""llama.cpp GGUF evaluation on GSM8K (resumable), with the same prompt and cap as scripts/run_gsm8k.py.

Differences from the original runner (kept as run_gguf_v1_cap256.py):
  * the prompt is the HF tokenizer's chat template output (the original used a different system message,
    "You are a helpful assistant." instead of Qwen's default system prompt, so the engine comparison also
    changed the prompt)
  * the generation cap comes from configs/manifest.yaml (1024), not 256
  * every item records finish_reason ("stop" | "length") and n_new_tokens
  * stores the robust v2 score plus the original v1 score

Usage: PYTHONPATH=src python scripts/run_gguf.py --model <gguf path> --model-name <name> [--run-id <id>] [--n-items 200]
"""
import argparse
import time

from datasets import load_dataset
from llama_cpp import Llama
from transformers import AutoTokenizer

from flipgate.health import run_health
from flipgate.manifest import load_manifest
from flipgate.scorers.gsm8k import GSM8KScorer, GSM8KScorerV1
from flipgate.store import ResultsStore, generate_run_id

TOKENIZER = "data/models/Qwen--Qwen2.5-3B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--manifest", default="configs/manifest.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    cap = args.max_new_tokens or manifest["sampling"]["max_tokens"]
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    items = [{"question": it["question"], "answer": it["answer"]} for it in gsm8k][:args.n_items]
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    fmt = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)

    store = ResultsStore("data/results")
    run_id = args.run_id or generate_run_id(args.model_name, "llama_cpp", "gsm8k", 1)
    done = {it["item_id"] for it in store.iter_items(run_id, "gsm8k")} if args.run_id else set()
    todo = [(i, it) for i, it in enumerate(items) if f"item_{i:04d}" not in done]
    print(f"run {run_id}: {len(done)} done, {len(todo)} to do, cap {cap}", flush=True)
    if not todo:
        return

    llm = Llama(model_path=args.model, n_ctx=4096, n_batch=512, n_gpu_layers=args.n_gpu_layers,
                n_threads=6, verbose=False)
    v2, v1 = GSM8KScorer(), GSM8KScorerV1()
    start = time.time()
    for n, (i, item) in enumerate(todo, 1):
        out = llm(fmt(item["question"]), max_tokens=cap, temperature=0.0, echo=False)
        ch = out["choices"][0]
        text, reason = ch["text"], ch.get("finish_reason") or "stop"
        ref = {"answer": item["answer"]}
        store.append_item(run_id, "gsm8k", f"item_{i:04d}", item["question"], text,
                          v2.score(item["question"], text, ref),
                          metadata={"finish_reason": reason, "n_new_tokens": out["usage"]["completion_tokens"],
                                    "score_v1": v1.score(item["question"], text, ref)})
        if n % 10 == 0:
            print(f"progress {len(done) + n}/{len(items)}", flush=True)

    rows = list(store.iter_items(run_id, "gsm8k"))
    health = run_health(rows, "gsm8k")
    acc = sum(1 for r in rows if r["score"] == 1.0) / len(rows)
    store.append_run_metadata(run_id, args.model_name, "llama_cpp", "0.3.35", "gsm8k", manifest["manifest_version"],
                              batch_size=1, extra={"accuracy": acc, "num_items": len(rows), "max_new_tokens": cap,
                                                   "scorer_version": GSM8KScorer.version,
                                                   "truncated_rate": health.truncated_rate,
                                                   "elapsed": time.time() - start})
    print(f"done: accuracy {acc:.1%}, truncated {health.truncated_rate:.1%}, run {run_id}", flush=True)


if __name__ == "__main__":
    main()

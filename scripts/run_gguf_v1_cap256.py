"""llama.cpp GGUF evaluation (resumable).

Runs GSM8K on a GGUF model via llama-cpp-python.
Usage: PYTHONPATH=src python scripts/run_gguf.py --model <path> --model-name <name> [--run-id <id>]
"""
import argparse
import json
import time
from datasets import load_dataset
from llama_cpp import Llama

from flipgate.scorers.gsm8k import GSM8KScorer
from flipgate.store import ResultsStore, generate_run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-items", type=int, default=200)
    args = parser.parse_args()

    print("Loading GSM8K...", flush=True)
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    items = [{"question": it["question"], "answer": it["answer"]} for it in gsm8k][:args.n_items]
    print(f"Using {len(items)} items", flush=True)

    store = ResultsStore("data/results")
    run_id = args.run_id or generate_run_id(args.model_name, "llama_cpp", "gsm8k", 1)
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

    print(f"Loading {args.model}...", flush=True)
    llm = Llama(model_path=args.model, n_ctx=4096, n_batch=512,
                n_gpu_layers=-1, n_threads=6, verbose=False)
    print("Model loaded", flush=True)

    scorer = GSM8KScorer()
    correct = sum(1 for it in store.iter_items(run_id, "gsm8k") if it["score"] == 1.0)
    start = time.time()
    # Qwen chat template for llama.cpp (no built-in template for GGUF)
    tmpl = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")

    for n, (i, item) in enumerate(todo, 1):
        out = llm(tmpl.format(prompt=item["question"]),
                  max_tokens=256, temperature=0.0, echo=False)
        response = out["choices"][0]["text"]
        score = scorer.score(item["question"], response, {"answer": item["answer"]})
        if score == 1.0:
            correct += 1
        store.append_item(run_id, "gsm8k", f"item_{i:04d}",
                          item["question"], response, score)
        if n % 20 == 0:
            done_total = len(done) + n
            print(f"Progress: {done_total}/{len(items)}, acc: {correct/done_total:.1%}", flush=True)

    elapsed = time.time() - start
    total = len(items)
    accuracy = correct / total
    store.append_run_metadata(
        run_id, args.model_name, "llama_cpp", "0.3.35", "gsm8k", "test",
        batch_size=1, extra={"accuracy": accuracy, "num_items": total, "elapsed": elapsed},
    )
    print(f"Done! {accuracy:.1%} ({correct}/{total}) in {elapsed:.1f}s", flush=True)
    print(f"Run ID: {run_id}", flush=True)


if __name__ == "__main__":
    main()

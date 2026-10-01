"""BFCL function-calling evaluation (resumable).

Scores executable-call rate: 1.0 iff the model emits the expected function
name with all required arguments (JSON {"name": ..., "arguments": {...}}).
Usage: PYTHONPATH=src python scripts/run_bfcl.py --model <path> --model-name <name> [--run-id <id>] [--file simple] [--n-items 399]
"""
import argparse
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flipgate.scorers.bfcl import BFCLScorer
from flipgate.store import ResultsStore, generate_run_id

BFCL_DIR = "/home/raihan/.cache/huggingface/hub/datasets--gorilla-llm--Berkeley-Function-Calling-Leaderboard/snapshots/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
FILES = {"simple": "BFCL_v3_simple.json", "multiple": "BFCL_v3_multiple.json",
         "parallel": "BFCL_v3_parallel.json",
         "exec_simple": "BFCL_v3_exec_simple.json",
         "exec_multiple": "BFCL_v3_exec_multiple.json"}

SYSTEM = ("You are a function-calling assistant. Given the available functions "
          "and the user request, reply with ONLY a JSON object like "
          '{"name": "<function>", "arguments": {...}} with all required arguments.')


def load_items(which: str, n: int | None):
    items = []
    with open(f"{BFCL_DIR}/{FILES[which]}") as f:
        for line in f:
            items.append(json.loads(line))
            if n is not None and len(items) >= n:
                break
    return items


def build_prompt(item: dict) -> tuple[str, dict]:
    funcs = item["function"]
    spec = json.dumps([{"name": fn["name"], "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {})} for fn in funcs])
    turns = item["question"]
    user_text = " ".join(m.get("content", "") for turn in turns for m in turn
                         if m.get("role") == "user")
    prompt = f"Available functions:\n{spec}\n\nUser request: {user_text}"
    ref = {"function": funcs}
    if item.get("ground_truth"):
        ref["ground_truth"] = item["ground_truth"]
    return prompt, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--file", default="simple", choices=list(FILES))
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    items = load_items(args.file, args.n_items)
    print(f"Loaded {len(items)} BFCL/{args.file} items", flush=True)

    store = ResultsStore("data/results")
    run_id = args.run_id or generate_run_id(args.model_name, "hf_generate", f"bfcl_{args.file}", 1)
    done = set()
    if args.run_id:
        done = {it["item_id"] for it in store.iter_items(run_id, f"bfcl_{args.file}")}
        print(f"Resuming {run_id}: {len(done)} done", flush=True)
    else:
        print(f"Run ID: {run_id}", flush=True)

    todo = [(i, it) for i, it in enumerate(items) if it["id"] not in done]
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

    scorer = BFCLScorer()
    dskey = f"bfcl_{args.file}"
    correct = sum(1 for it in store.iter_items(run_id, dskey) if it["score"] == 1.0)
    start = time.time()
    for n, (i, item) in enumerate(todo, 1):
        prompt, ref = build_prompt(item)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                           max_length=4096 - 256).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True)
        score = scorer.score(prompt, response, ref)
        if score == 1.0:
            correct += 1
        store.append_item(run_id, dskey, item["id"], prompt[:400],
                          response[:800], score)
        if n % 20 == 0:
            done_total = len(done) + n
            print(f"Progress: {done_total}/{len(items)}, exec-rate: {correct/done_total:.1%}",
                  flush=True)

    elapsed = time.time() - start
    total = len(items)
    rate = correct / total
    store.append_run_metadata(run_id, args.model_name, "hf_generate", "5.17.0",
                              dskey, "test", batch_size=1,
                              extra={"accuracy": rate, "num_items": total, "elapsed": elapsed})
    print(f"Done! Executable-call rate: {rate:.1%} ({correct}/{total}) in {elapsed:.1f}s", flush=True)
    print(f"Run ID: {run_id}", flush=True)


if __name__ == "__main__":
    main()

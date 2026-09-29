"""IFEval evaluation runner (resumable).

Runs all 541 IFEval prompts on a model with the rule-based scorer.
Resumes from existing items in the store, so interrupted runs can continue.
Usage: PYTHONPATH=src python scripts/run_ifeval.py --model <path> --model-name <name>
"""
import argparse
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flipgate.scorers.ifeval import IFEvalScorer
from flipgate.store import ResultsStore, generate_run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-id", default=None, help="Resume existing run")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    print("Loading IFEval dataset...", flush=True)
    path = "/home/raihan/.cache/huggingface/hub/datasets--google--IFEval/snapshots/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl"
    with open(path) as f:
        all_items = [json.loads(line) for line in f]
    print(f"Loaded {len(all_items)} items", flush=True)

    store = ResultsStore("data/results")
    run_id = args.run_id or generate_run_id(args.model_name, "hf_generate", "ifeval", 1)
    done = set()
    if args.run_id:
        done = {it["item_id"] for it in store.iter_items(run_id, "ifeval")}
        print(f"Resuming {run_id}: {len(done)} items already done", flush=True)
    else:
        print(f"Run ID: {run_id}", flush=True)

    todo = [(i, it) for i, it in enumerate(all_items) if f"item_{i:04d}" not in done]
    print(f"Remaining: {len(todo)} items", flush=True)
    if not todo:
        print("Nothing to do.", flush=True)
        return

    print(f"Loading {args.model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if "bf16" in args.model_name else torch.float16,
        device_map="cuda", pad_token_id=tokenizer.pad_token_id,
    )
    print("Model loaded", flush=True)

    scorer = IFEvalScorer()
    correct = sum(1 for it in store.iter_items(run_id, "ifeval") if it["score"] == 1.0)
    start = time.time()

    for i, item in todo:
        messages = [{"role": "user", "content": item["prompt"]}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                           max_length=4096 - args.max_tokens).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        reference = {"instruction_id_list": item["instruction_id_list"], "kwargs": item["kwargs"]}
        score = scorer.score(item["prompt"], response, reference)
        if score == 1.0:
            correct += 1
        store.append_item(run_id, "ifeval", f"item_{i:04d}",
                          item["prompt"][:500], response[:2000], score)
        done_count = len(done) + (todo.index((i, item)) + 1)
        if done_count % 20 == 0:
            print(f"Progress: {done_count}/{len(all_items)}, acc: {correct/done_count:.1%}", flush=True)

    elapsed = time.time() - start
    total = len(all_items)
    accuracy = correct / total
    store.append_run_metadata(
        run_id, args.model_name, "hf_generate", "5.17.0", "ifeval", "test",
        batch_size=1, extra={"accuracy": accuracy, "num_items": total, "elapsed": elapsed},
    )
    print(f"Done! {accuracy:.1%} ({correct}/{total}) in {elapsed:.1f}s", flush=True)
    print(f"Run ID: {run_id}", flush=True)


if __name__ == "__main__":
    main()

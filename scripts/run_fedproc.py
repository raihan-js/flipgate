"""FedProc registry-hallucination evaluation (resumable).

Prompts the model with a clause topic excerpt and asks for the applicable
FAR/DFARS clause number. Scores 1.0 iff every cited number exists in the
registry (no hallucination), else 0.0. Uses only the real-FAR slice
(155 records); Claude-written synthetic records stay out.
Usage: PYTHONPATH=src python scripts/run_fedproc.py --model <path> --model-name <name> [--run-id <id>]
"""
import argparse
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flipgate.scorers.fedproc import FedProcRegistryScorer
from flipgate.store import ResultsStore, generate_run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    print("Loading registry and items...", flush=True)
    registry = set(json.load(open("data/eval/far_registry.json")))
    items = json.load(open("data/eval/fedproc_items.json"))
    print(f"Registry: {len(registry)} clauses, Items: {len(items)}", flush=True)

    store = ResultsStore("data/results")
    run_id = args.run_id or generate_run_id(args.model_name, "hf_generate", "fedproc", 1)
    done = set()
    if args.run_id:
        done = {it["item_id"] for it in store.iter_items(run_id, "fedproc")}
        print(f"Resuming {run_id}: {len(done)} done", flush=True)
    else:
        print(f"Run ID: {run_id}", flush=True)

    todo = [it for it in items if it["item_id"] not in done]
    print(f"Remaining: {len(todo)}", flush=True)
    if not todo:
        print("Nothing to do.", flush=True)
        return

    print(f"Loading {args.model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda",
        pad_token_id=tokenizer.pad_token_id,
    )
    print("Model loaded", flush=True)

    scorer = FedProcRegistryScorer(registry=registry)
    correct = sum(1 for it in store.iter_items(run_id, "fedproc") if it["score"] == 1.0)
    start = time.time()
    true_number = {it["item_id"]: it["item_id"].split(":")[1] for it in items}

    for n, item in enumerate(todo, 1):
        excerpt = item["text"][:300]
        prompt = (
            "A federal contract needs a clause covering the following topic:\n\n"
            f"{excerpt}\n\n"
            "Which FAR or DFARS clause number applies? "
            "Reply with the clause number in the format FAR XX.XXX-X."
        )
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                           max_length=4096 - 128).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        score = scorer.score(prompt, response, {})
        if score == 1.0:
            correct += 1
        hallucinated = scorer.get_hallucinated_clauses(response)
        store.append_item(run_id, "fedproc", item["item_id"], prompt[:300], response[:500], score,
                          metadata={"true_number": true_number[item["item_id"]],
                                    "hallucinated": hallucinated})
        if n % 20 == 0:
            done_total = len(done) + n
            print(f"Progress: {done_total}/{len(items)}, no-hallucination rate: {correct/done_total:.1%}", flush=True)

    elapsed = time.time() - start
    total = len(items)
    rate = correct / total
    store.append_run_metadata(
        run_id, args.model_name, "hf_generate", "5.17.0", "fedproc", "test",
        batch_size=1, extra={"accuracy": rate, "num_items": total, "elapsed": elapsed},
    )
    print(f"Done! No-hallucination rate: {rate:.1%} ({correct}/{total}) in {elapsed:.1f}s", flush=True)
    print(f"Run ID: {run_id}", flush=True)


if __name__ == "__main__":
    main()

"""Noise floor measurement: bf16-vs-bf16 flip rates at different batch sizes.

This script measures the inherent nondeterminism of LLM inference by running
the same model multiple times on the same inputs and counting answer flips.

Under greedy decoding (temp=0), random seeds don't matter. The real noise
comes from batched kernel operations and floating-point reduction order.
"""

import json
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

from flipgate.scorers.gsm8k import GSM8KScorer
from flipgate.store import ResultsStore, generate_run_id
from flipgate.stats.bootstrap import compute_flip_rate, paired_bootstrap_ci
from flipgate.stats.mcnemar import count_flips, mcnemar_test

console = Console()


def run_batch_inference(
    model,
    tokenizer,
    items: list[dict],
    batch_size: int,
    max_new_tokens: int = 256,
) -> list[str]:
    """Run batched inference on items."""
    all_responses = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        prompts = []
        for item in batch:
            messages = [{"role": "user", "content": item["question"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(formatted)
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        input_len = inputs["input_ids"].shape[1]
        for output in outputs:
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            all_responses.append(response)
    
    return all_responses


def measure_noise_floor(
    model_path: str,
    items: list[dict],
    batch_sizes: list[int],
    num_runs: int = 3,
    results_dir: str = "data/results",
):
    """Measure noise floor at different batch sizes."""
    
    console.print(f"\n[bold cyan]Loading model from {model_path}...[/bold cyan]")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda",
        pad_token_id=tokenizer.pad_token_id,
    )
    console.print(f"✓ Model loaded on {model.device}\n")
    
    scorer = GSM8KScorer()
    store = ResultsStore(results_dir)
    
    results = {}
    
    for batch_size in batch_sizes:
        console.print(f"[bold]Batch size {batch_size}[/bold]")
        run_ids = []
        accuracies = []
        
        for run_idx in range(num_runs):
            run_id = generate_run_id(
                "Qwen2.5-3B-bf16", "hf_generate", "gsm8k", batch_size
            )
            run_ids.append(run_id)
            
            start = time.time()
            responses = run_batch_inference(model, tokenizer, items, batch_size)
            elapsed = time.time() - start
            
            correct = 0
            for item, response in zip(items, responses):
                score = scorer.score(item["question"], response, {"answer": item["answer"]})
                if score == 1.0:
                    correct += 1
                store.append_item(
                    run_id, "gsm8k", item["item_id"],
                    item["question"], response, score
                )
            
            acc = correct / len(items)
            accuracies.append(acc)
            store.append_run_metadata(
                run_id, "Qwen2.5-3B-Instruct", "hf_generate", "5.17.0",
                "gsm8k", "test", batch_size=batch_size,
                extra={"accuracy": acc, "num_items": len(items), "elapsed": elapsed}
            )
            
            console.print(f"  Run {run_idx + 1}/{num_runs}: acc={acc:.0%}, time={elapsed:.1f}s")
        
        # Compare all pairs of runs
        flip_rates = []
        for i in range(len(run_ids)):
            for j in range(i + 1, len(run_ids)):
                items_i = {it["item_id"]: it for it in store.iter_items(run_ids[i], "gsm8k")}
                items_j = {it["item_id"]: it for it in store.iter_items(run_ids[j], "gsm8k")}
                
                common = sorted(set(items_i.keys()) & set(items_j.keys()))
                correct_i = [items_i[k]["score"] == 1.0 for k in common]
                correct_j = [items_j[k]["score"] == 1.0 for k in common]
                
                flips = count_flips(correct_i, correct_j)
                flip_rate = compute_flip_rate(correct_i, correct_j, direction="any")
                flip_rates.append(flip_rate)
                
                console.print(
                    f"    {run_ids[i][-8:]} vs {run_ids[j][-8:]}: "
                    f"r2w={flips['right_to_wrong']}, w2r={flips['wrong_to_right']}, "
                    f"rate={flip_rate:.4f}"
                )
        
        mean_rate = sum(flip_rates) / len(flip_rates) if flip_rates else 0
        results[batch_size] = {
            "run_ids": run_ids,
            "accuracies": accuracies,
            "flip_rates": flip_rates,
            "mean_flip_rate": mean_rate,
        }
        
        console.print(f"  [cyan]Mean flip rate: {mean_rate:.4f}[/cyan]\n")
    
    # Summary table
    table = Table(title="Noise Floor Summary (bf16-vs-bf16)")
    table.add_column("Batch Size", style="cyan")
    table.add_column("Mean Flip Rate", style="green")
    table.add_column("Interpretation")
    
    for bs, data in results.items():
        rate = data["mean_flip_rate"]
        if rate == 0:
            interp = "Deterministic"
        elif rate < 0.01:
            interp = "Very low noise"
        elif rate < 0.05:
            interp = "Low noise"
        else:
            interp = "Significant noise"
        table.add_row(str(bs), f"{rate:.4f}", interp)
    
    console.print(table)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen--Qwen2.5-3B-Instruct")
    parser.add_argument("--items", type=int, default=30)
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    
    with open("data/eval/gsm8k_sample.json") as f:
        all_items = json.load(f)
    items = all_items[:args.items]
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    console.print(f"\n[bold]Noise Floor Measurement[/bold]")
    console.print(f"Items: {len(items)}, Batch sizes: {batch_sizes}, Runs: {args.runs}\n")
    
    results = measure_noise_floor(
        model_path=args.model,
        items=items,
        batch_sizes=batch_sizes,
        num_runs=args.runs,
    )
    
    console.print("\n[bold green]✓ Noise floor measurement complete[/bold green]")

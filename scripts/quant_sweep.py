"""Quantization sweep: compare bf16 vs AWQ vs GPTQ-Int4.

This script runs the same evaluation on different quantization methods
and measures per-item answer flips to detect regressions.
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


def run_evaluation(
    model_path: str,
    model_name: str,
    items: list[dict],
    batch_size: int = 1,
    results_dir: str = "data/results",
):
    """Run evaluation on a model and return results."""
    
    console.print(f"\n[bold cyan]Loading {model_name}...[/bold cyan]")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - transformers auto-detects quantization from config.json
    model_kwargs = {
        "device_map": "cuda",
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    if "bf16" in model_name:
        model_kwargs["dtype"] = torch.bfloat16
    else:
        # AWQ/GPTQ models auto-detect their quantization config
        model_kwargs["torch_dtype"] = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    # Get model size
    model_size = sum(p.numel() for p in model.parameters()) / 1e9
    
    scorer = GSM8KScorer()
    store = ResultsStore(results_dir)
    
    run_id = generate_run_id(model_name, "hf_generate", "gsm8k", batch_size)
    
    console.print(f"✓ Model loaded ({model_size:.2f}B params)")
    
    # Run inference
    start = time.time()
    correct = 0
    
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
                max_new_tokens=256,
                do_sample=False,
            )
        
        input_len = inputs["input_ids"].shape[1]
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            item = batch[j]
            
            score = scorer.score(item["question"], response, {"answer": item["answer"]})
            if score == 1.0:
                correct += 1
            
            store.append_item(
                run_id, "gsm8k", item["item_id"],
                item["question"], response, score
            )
    
    elapsed = time.time() - start
    accuracy = correct / len(items)
    
    store.append_run_metadata(
        run_id, model_name, "hf_generate", "5.17.0",
        "gsm8k", "test", batch_size=batch_size,
        extra={
            "accuracy": accuracy,
            "num_items": len(items),
            "elapsed": elapsed,
            "model_size_gb": model_size,
        }
    )
    
    console.print(f"  Accuracy: {accuracy:.0%} ({correct}/{len(items)})")
    console.print(f"  Time: {elapsed:.1f}s")
    
    return {
        "run_id": run_id,
        "model_name": model_name,
        "accuracy": accuracy,
        "elapsed": elapsed,
        "model_size": model_size,
    }


def compare_runs(baseline_run: dict, candidate_run: dict, items: list[dict]):
    """Compare two runs and compute flip statistics."""
    store = ResultsStore("data/results")
    
    baseline_items = {it["item_id"]: it for it in store.iter_items(baseline_run["run_id"], "gsm8k")}
    candidate_items = {it["item_id"]: it for it in store.iter_items(candidate_run["run_id"], "gsm8k")}
    
    common = sorted(set(baseline_items.keys()) & set(candidate_items.keys()))
    correct_b = [baseline_items[k]["score"] == 1.0 for k in common]
    correct_c = [candidate_items[k]["score"] == 1.0 for k in common]
    
    flips = count_flips(correct_b, correct_c)
    r2w_rate = compute_flip_rate(correct_b, correct_c, direction="right_to_wrong")
    w2r_rate = compute_flip_rate(correct_b, correct_c, direction="wrong_to_right")
    any_rate = compute_flip_rate(correct_b, correct_c, direction="any")
    
    mcnemar = mcnemar_test(correct_b, correct_c)
    
    # Bootstrap CI for right-to-wrong flip rate
    bootstrap = paired_bootstrap_ci(correct_b, correct_c, direction="right_to_wrong", seed=42)
    
    return {
        "right_to_wrong": flips["right_to_wrong"],
        "wrong_to_right": flips["wrong_to_right"],
        "total_flips": flips["total_flips"],
        "r2w_rate": r2w_rate,
        "w2r_rate": w2r_rate,
        "any_rate": any_rate,
        "mcnemar_p": mcnemar.p_value,
        "mcnemar_significant": mcnemar.significant,
        "bootstrap_ci": (bootstrap.ci_lower, bootstrap.ci_upper),
    }


def run_quantization_sweep(
    items: list[dict],
    results_dir: str = "data/results",
):
    """Run quantization sweep comparing bf16 vs AWQ vs GPTQ."""
    
    models = [
        ("data/models/Qwen--Qwen2.5-3B-Instruct", "Qwen2.5-3B-bf16"),
        ("data/models/Qwen--Qwen2.5-3B-Instruct-AWQ", "Qwen2.5-3B-AWQ"),
        ("data/models/Qwen--Qwen2.5-3B-Instruct-GPTQ-Int4", "Qwen2.5-3B-GPTQ-Int4"),
    ]
    
    results = []
    for model_path, model_name in models:
        result = run_evaluation(model_path, model_name, items, batch_size=1, results_dir=results_dir)
        results.append(result)
    
    # Compare all quantized models against bf16 baseline
    baseline = results[0]
    comparisons = []
    
    for candidate in results[1:]:
        comp = compare_runs(baseline, candidate, items)
        comparisons.append((candidate, comp))
    
    # Summary table
    console.print("\n" + "=" * 80)
    console.print("[bold]Quantization Sweep Results[/bold]")
    console.print("=" * 80)
    
    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Size (GB)", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Time (s)", justify="right")
    
    for result in results:
        table.add_row(
            result["model_name"],
            f"{result['model_size']:.2f}",
            f"{result['accuracy']:.0%}",
            f"{result['elapsed']:.1f}",
        )
    
    console.print(table)
    
    # Flip analysis table
    table = Table(title="Flip Analysis (vs bf16 baseline)")
    table.add_column("Candidate", style="cyan")
    table.add_column("Right→Wrong", justify="right")
    table.add_column("Wrong→Right", justify="right")
    table.add_column("R→W Rate", justify="right")
    table.add_column("McNemar p", justify="right")
    table.add_column("Significant?", justify="center")
    
    for candidate, comp in comparisons:
        sig = "[red]YES[/red]" if comp["mcnemar_significant"] else "no"
        table.add_row(
            candidate["model_name"],
            str(comp["right_to_wrong"]),
            str(comp["wrong_to_right"]),
            f"{comp['r2w_rate']:.4f}",
            f"{comp['mcnemar_p']:.4f}",
            sig,
        )
    
    console.print(table)
    
    return results, comparisons


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=30)
    args = parser.parse_args()
    
    with open("data/eval/gsm8k_sample.json") as f:
        all_items = json.load(f)
    items = all_items[:args.items]
    
    console.print(f"\n[bold]Quantization Sweep[/bold]")
    console.print(f"Items: {len(items)}")
    console.print(f"Models: bf16, AWQ, GPTQ-Int4\n")
    
    results, comparisons = run_quantization_sweep(items)
    
    console.print("\n[bold green]✓ Quantization sweep complete[/bold green]")

"""Evaluation runner: ties engines, scorers, and store together."""

import json
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress

from flipgate.manifest import load_manifest, get_model_config, get_dataset_config
from flipgate.store import ResultsStore, generate_run_id, compute_manifest_sha
from flipgate.scorers.gsm8k import GSM8KScorer
from flipgate.scorers.ifeval import IFEvalScorer
from flipgate.scorers.fedproc import FedProcRegistryScorer

console = Console()


def get_scorer(scorer_name: str, **kwargs):
    """Get scorer instance by name."""
    scorers = {
        "gsm8k": GSM8KScorer,
        "ifeval": IFEvalScorer,
        "fedproc_registry": FedProcRegistryScorer,
    }
    if scorer_name not in scorers:
        raise ValueError(f"Unknown scorer: {scorer_name}")
    return scorers[scorer_name](**kwargs)


def load_eval_dataset(dataset_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load evaluation dataset from HuggingFace.
    
    Returns list of dicts with keys: item_id, prompt, reference
    """
    source = dataset_config["source"]
    subset = dataset_config.get("subset")
    split = dataset_config.get("split", "test")
    num_items = dataset_config.get("num_items")
    
    console.print(f"Loading {source} ({split} split)...")
    
    load_kwargs = {"split": split}
    if subset:
        load_kwargs["name"] = subset
    
    ds = load_dataset(source, **load_kwargs)
    
    items = []
    for i, row in enumerate(ds):
        if num_items and i >= num_items:
            break
        
        # Extract prompt and reference based on dataset
        if "gsm8k" in source.lower():
            prompt = row["question"]
            reference = {"answer": row["answer"]}
        elif "ifeval" in source.lower():
            prompt = row["prompt"]
            reference = {
                "instruction_id_list": row.get("instruction_id_list", []),
                "kwargs": row.get("kwargs", []),
            }
        else:
            prompt = row.get("prompt", row.get("question", str(row)))
            reference = row
        
        items.append({
            "item_id": f"{i:05d}",
            "prompt": prompt,
            "reference": reference,
        })
    
    console.print(f"Loaded {len(items)} items")
    return items


def apply_chat_template(
    tokenizer,
    prompt: str,
    model_config: dict[str, Any],
) -> str:
    """Apply chat template to format prompt for the model."""
    messages = [{"role": "user", "content": prompt}]
    
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def run_evaluation(
    model_key: str,
    engine_name: str,
    dataset_key: str,
    manifest_path: Path | None = None,
    results_dir: str = "data/results",
    batch_size: int = 1,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Run a full evaluation.
    
    Args:
        model_key: Key from manifest (e.g., "base", "awq")
        engine_name: Engine to use ("hf_generate", "vllm", "llama_cpp")
        dataset_key: Dataset key from manifest (e.g., "gsm8k", "ifeval")
        manifest_path: Path to manifest.yaml
        results_dir: Directory for results
        batch_size: Batch size for inference
        max_items: Limit number of items (for testing)
    
    Returns:
        Dictionary with run_id, accuracy, flip_rate, etc.
    """
    # Load manifest
    manifest = load_manifest(manifest_path)
    model_config = get_model_config(manifest, model_key)
    dataset_config = get_dataset_config(manifest, dataset_key)
    
    # Generate run ID
    run_id = generate_run_id(model_key, engine_name, dataset_key, batch_size)
    console.print(f"\n[bold cyan]Run ID:[/bold cyan] {run_id}")
    
    # Initialize store
    store = ResultsStore(results_dir)
    manifest_sha = compute_manifest_sha(
        manifest_path or Path(__file__).parent.parent / "configs" / "manifest.yaml"
    )
    
    # Load dataset
    items = load_eval_dataset(dataset_config)
    if max_items:
        items = items[:max_items]
    
    # Initialize engine
    console.print(f"Loading model {model_config['id']} with {engine_name}...")
    
    if engine_name == "hf_generate":
        from flipgate.engines.hf_engine import HFGenerateEngine
        engine = HFGenerateEngine(
            model_id=model_config["id"],
            dtype=model_config.get("dtype", "bfloat16"),
        )
    elif engine_name == "vllm":
        from flipgate.engines.vllm_engine import VLLMEngine
        engine = VLLMEngine(
            model_id=model_config["id"],
            dtype=model_config.get("dtype", "auto"),
            quantization=model_config.get("quantization"),
        )
    elif engine_name == "llama_cpp":
        from flipgate.engines.llamacpp_engine import LlamaCppEngine
        engine = LlamaCppEngine(
            model_path=model_config.get("model_path", ""),
        )
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
    
    # Get scorer
    scorer = get_scorer(dataset_config["scorer"])
    
    # Apply chat template to prompts
    if hasattr(engine, '_tokenizer') and engine._tokenizer is not None:
        tokenizer = engine._tokenizer
    elif hasattr(engine, '_load_model'):
        engine._load_model()
        tokenizer = getattr(engine, '_tokenizer', None)
    else:
        tokenizer = None
    
    formatted_prompts = []
    for item in items:
        if tokenizer:
            formatted = apply_chat_template(tokenizer, item["prompt"], model_config)
        else:
            formatted = item["prompt"]
        formatted_prompts.append(formatted)
    
    # Run inference
    console.print(f"Generating responses for {len(items)} items...")
    start_time = time.time()
    
    responses = engine.generate(
        formatted_prompts,
        max_tokens=manifest["sampling"]["max_tokens"],
        temperature=manifest["sampling"]["temperature"],
        batch_size=batch_size,
    )
    
    elapsed = time.time() - start_time
    console.print(f"Generated {len(responses)} responses in {elapsed:.1f}s")
    
    # Score and store
    correct_count = 0
    with Progress() as progress:
        task = progress.add_task("Scoring...", total=len(items))
        
        for i, (item, response) in enumerate(zip(items, responses)):
            score = scorer.score(item["prompt"], response, item["reference"])
            if score == 1.0:
                correct_count += 1
            
            store.append_item(
                run_id=run_id,
                dataset=dataset_key,
                item_id=item["item_id"],
                prompt=item["prompt"],
                response=response,
                score=score,
            )
            progress.update(task, advance=1)
    
    accuracy = correct_count / len(items) if items else 0.0
    
    # Store metadata
    engine_info = engine.get_engine_info()
    store.append_run_metadata(
        run_id=run_id,
        model=model_config["id"],
        engine=engine_name,
        engine_version=engine_info.get("version", "unknown"),
        dataset=dataset_key,
        manifest_sha=manifest_sha,
        batch_size=batch_size,
        extra={
            "num_items": len(items),
            "accuracy": accuracy,
            "elapsed_seconds": elapsed,
            "tokens_per_second": len(responses) / elapsed if elapsed > 0 else 0,
        },
    )
    
    console.print(f"\n[bold green]Accuracy:[/bold green] {accuracy:.2%} ({correct_count}/{len(items)})")
    
    return {
        "run_id": run_id,
        "accuracy": accuracy,
        "num_items": len(items),
        "elapsed_seconds": elapsed,
    }


def run_noise_floor(
    model_key: str = "base",
    engine_name: str = "hf_generate",
    dataset_key: str = "gsm8k",
    batch_sizes: list[int] = None,
    num_runs: int = 3,
    manifest_path: Path | None = None,
    results_dir: str = "data/results",
    max_items: int = 100,
) -> dict[str, Any]:
    """Run noise floor measurement.
    
    Runs the same model multiple times at different batch sizes to measure
    inherent nondeterminism.
    """
    from flipgate.stats import compute_flip_rate
    
    if batch_sizes is None:
        batch_sizes = [1, 8, 32]
    
    console.print(f"\n[bold cyan]Noise Floor Measurement[/bold cyan]")
    console.print(f"Model: {model_key}, Engine: {engine_name}")
    console.print(f"Batch sizes: {batch_sizes}, Runs per config: {num_runs}")
    
    results = {}
    
    for batch_size in batch_sizes:
        console.print(f"\n[bold]Batch size {batch_size}[/bold]")
        run_ids = []
        
        for run_idx in range(num_runs):
            console.print(f"  Run {run_idx + 1}/{num_runs}")
            result = run_evaluation(
                model_key=model_key,
                engine_name=engine_name,
                dataset_key=dataset_key,
                manifest_path=manifest_path,
                results_dir=results_dir,
                batch_size=batch_size,
                max_items=max_items,
            )
            run_ids.append(result["run_id"])
        
        # Compare runs to measure flip rates
        store = ResultsStore(results_dir)
        flip_rates = []
        
        for i in range(len(run_ids)):
            for j in range(i + 1, len(run_ids)):
                items_i = {item["item_id"]: item for item in store.iter_items(run_ids[i], dataset_key)}
                items_j = {item["item_id"]: item for item in store.iter_items(run_ids[j], dataset_key)}
                
                common = sorted(set(items_i.keys()) & set(items_j.keys()))
                correct_i = [items_i[k]["score"] == 1.0 for k in common]
                correct_j = [items_j[k]["score"] == 1.0 for k in common]
                
                flip_rate = compute_flip_rate(correct_i, correct_j, direction="any")
                flip_rates.append(flip_rate)
                console.print(f"  Flips {run_ids[i][-8:]} vs {run_ids[j][-8:]}: {flip_rate:.4f}")
        
        results[batch_size] = {
            "run_ids": run_ids,
            "flip_rates": flip_rates,
            "mean_flip_rate": sum(flip_rates) / len(flip_rates) if flip_rates else 0,
        }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FlipGate evaluation")
    parser.add_argument("--model", default="base", help="Model key from manifest")
    parser.add_argument("--engine", default="hf_generate", help="Engine name")
    parser.add_argument("--dataset", default="gsm8k", help="Dataset key")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--noise-floor", action="store_true")
    
    args = parser.parse_args()
    
    if args.noise_floor:
        run_noise_floor(
            model_key=args.model,
            engine_name=args.engine,
            dataset_key=args.dataset,
            max_items=args.max_items or 100,
        )
    else:
        run_evaluation(
            model_key=args.model,
            engine_name=args.engine,
            dataset_key=args.dataset,
            batch_size=args.batch_size,
            max_items=args.max_items,
        )

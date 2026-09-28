#!/usr/bin/env python3
"""
vLLM noise floor measurement at different batch sizes.
Measures bf16-vs-bf16 flip rates to establish the baseline nondeterminism.
"""
import json
import time
from pathlib import Path
from vllm import LLM, SamplingParams
from flipgate.store import ResultsStore, generate_run_id
from flipgate.scorers.gsm8k import GSM8KScorer

def measure_vllm_noise_floor(model_path: str, dataset_path: str, batch_sizes: list[int], num_runs: int = 3):
    """
    Measure noise floor by running same model multiple times at different batch sizes.
    
    Args:
        model_path: Path to bf16 model
        dataset_path: Path to GSM8K dataset
        batch_sizes: List of batch sizes to test (e.g., [1, 8, 32])
        num_runs: Number of runs per batch size
    """
    store = ResultsStore("data/results")
    scorer = GSM8KScorer()
    
    # Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    prompts = [item["question"] for item in dataset]
    references = [{"answer": item["answer"]} for item in dataset]
    
    print(f"Loaded {len(prompts)} items from {dataset_path}")
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Running {num_runs} runs per batch size\n")
    
    # Sampling params for greedy decoding
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"{'='*60}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}")
        
        batch_results = []
        
        for run_idx in range(num_runs):
            print(f"\nRun {run_idx + 1}/{num_runs}...")
            
            # Initialize vLLM engine
            llm = LLM(
                model=model_path,
                dtype="bfloat16",
                max_model_len=2048,
                gpu_memory_utilization=0.85,  # Reduced from 0.9 to account for system overhead
                enforce_eager=True,  # Disable CUDA graphs to avoid flashinfer issues
            )
            
            # Generate
            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.time() - start_time
            
            responses = [output.outputs[0].text for output in outputs]
            
            # Score
            scores = []
            for prompt, response, reference in zip(prompts, responses, references):
                score = scorer.score(prompt, response, reference)
                scores.append(score)
            
            accuracy = sum(scores) / len(scores)
            print(f"  Accuracy: {accuracy:.1%} ({sum(scores)}/{len(scores)})")
            print(f"  Time: {elapsed:.2f}s")
            
            # Store results
            run_id = generate_run_id(
                "Qwen2.5-3B-bf16",
                "vllm",
                "gsm8k",
                batch_size,
            )
            
            # Store each item
            for idx, (prompt, response, score) in enumerate(zip(prompts, responses, scores)):
                store.append_item(
                    run_id=run_id,
                    dataset="gsm8k",
                    item_id=f"item_{idx:04d}",
                    prompt=prompt,
                    response=response,
                    score=score,
                )
            
            # Store metadata
            store.append_run_metadata(
                run_id=run_id,
                model="Qwen2.5-3B-Instruct",
                engine="vllm",
                engine_version="0.30.0",
                dataset="gsm8k",
                manifest_sha="vllm_noise_floor",
                batch_size=batch_size,
                extra={"elapsed_seconds": elapsed}
            )
            
            batch_results.append(run_id)
            
            # Free memory
            del llm
            import torch
            torch.cuda.empty_cache()
        
        results[batch_size] = batch_results
    
    # Analyze flip rates
    print(f"\n{'='*60}")
    print("NOISE FLOOR ANALYSIS")
    print(f"{'='*60}\n")
    
    for batch_size, run_ids in results.items():
        print(f"Batch size {batch_size}:")
        
        # Compare all pairs of runs
        flip_rates = []
        for i in range(len(run_ids)):
            for j in range(i + 1, len(run_ids)):
                items_i = list(store.iter_items(run_ids[i], "gsm8k"))
                items_j = list(store.iter_items(run_ids[j], "gsm8k"))
                
                scores_i = [item["score"] for item in items_i]
                scores_j = [item["score"] for item in items_j]
                
                # Count flips
                flips = sum(1 for s_i, s_j in zip(scores_i, scores_j) if s_i != s_j)
                flip_rate = flips / len(scores_i)
                flip_rates.append(flip_rate)
                
                print(f"  {run_ids[i][-8:]} vs {run_ids[j][-8:]}: {flips} flips ({flip_rate:.2%})")
        
        mean_flip_rate = sum(flip_rates) / len(flip_rates) if flip_rates else 0
        print(f"  Mean flip rate: {mean_flip_rate:.2%}\n")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen--Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset", default="data/eval/gsm8k_sample.json")
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    results = measure_vllm_noise_floor(
        model_path=args.model,
        dataset_path=args.dataset,
        batch_sizes=batch_sizes,
        num_runs=args.runs,
    )
    
    print("\n✓ Noise floor measurement complete")
    print(f"Results stored in data/results/")

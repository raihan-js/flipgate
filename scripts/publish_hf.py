#!/usr/bin/env python3
"""Export FlipGate results to HuggingFace dataset."""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict

def export_results():
    """Export all results to HuggingFace dataset format."""
    results_dir = Path("data/results")
    
    if not results_dir.exists():
        print("No results directory found")
        return None
    
    # Collect all run metadata and results
    all_runs = []
    
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        metadata_file = run_dir / "metadata.json"
        if not metadata_file.exists():
            continue
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Load all JSONL files in this run
        results = {}
        for jsonl_file in run_dir.glob("*.jsonl"):
            dataset_name = jsonl_file.stem
            items = []
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
            results[dataset_name] = items
        
        run_data = {
            "run_id": run_dir.name,
            "metadata": metadata,
            "results": results
        }
        all_runs.append(run_data)
    
    print(f"Found {len(all_runs)} runs")
    
    # Create dataset
    dataset = Dataset.from_list(all_runs)
    dataset_dict = DatasetDict({"train": dataset})
    
    return dataset_dict


def create_dataset_card():
    """Create a dataset card (README.md) for the dataset."""
    card = """---
license: mit
task_categories:
  - text-generation
tags:
  - llm-evaluation
  - quantization
  - model-comparison
  - flipgate
pretty_name: FlipGate Results
size_categories:
  - n<1K
---

# FlipGate Results

Per-item evaluation results from FlipGate, a release gate for quantised and re-served LLMs.

## Dataset Description

This dataset contains per-item evaluation results comparing different quantization methods (bf16, AWQ, GPTQ-Int4) on Qwen2.5-3B-Instruct.

Each run includes:
- **metadata.json**: Model config, engine version, dataset, batch size, accuracy
- **{dataset}.jsonl**: Per-item results with prompts, responses, and scores

## Key Findings

- **Noise floor**: HF generate is deterministic at batch_size=1 and 8 (0 flips)
- **Quantization sweep**: No statistically significant regressions detected with McNemar's test
- **bf16 baseline**: 30% accuracy on 30-item GSM8K sample
- **AWQ**: 27% accuracy, 3 right-to-wrong flips
- **GPTQ-Int4**: 33% accuracy, 0 right-to-wrong flips

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("raihan-js/flipgate-results")
run = dataset["train"][0]

print(run["metadata"])  # Model config, accuracy, etc.
print(run["results"]["gsm8k"])  # Per-item results
```

## Methodology

- **Model**: Qwen2.5-3B-Instruct (bf16, AWQ, GPTQ-Int4)
- **Engine**: HuggingFace Transformers generate()
- **Sampling**: temperature=0.0 (greedy decoding)
- **Scorer**: GSM8K exact numeric match
- **Statistics**: McNemar's test for paired comparisons

## Limitations

- Small sample size (30 items) limits statistical power
- Only HF generate engine tested (vLLM installation timed out)
- Noise floor measurement incomplete (needs vLLM batched kernels)

## Citation

```bibtex
@software{flipgate2026,
  author = {Raihan Sikder},
  title = {FlipGate: Release gate for quantised LLMs},
  year = {2026},
  url = {https://github.com/raihan-js/flipgate}
}
```
"""
    return card


def main():
    """Main export function."""
    print("Exporting FlipGate results to HuggingFace...")
    
    # Export results
    dataset_dict = export_results()
    if dataset_dict is None:
        return
    
    # Create dataset card
    card_content = create_dataset_card()
    
    # Push to HuggingFace
    repo_id = "raihan-js/flipgate-results"
    print(f"\nPushing to {repo_id}...")
    
    dataset_dict.push_to_hub(
        repo_id,
        private=False
    )
    
    # Add README separately
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    print(f"\n✓ Published to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()

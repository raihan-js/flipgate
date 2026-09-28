# FlipGate

A CLI and GitHub Action release gate for quantised and re-served LLMs. Counts per-item right-to-wrong answer flips against a measured bf16 noise floor, using paired statistics (McNemar, paired bootstrap) instead of aggregate accuracy.

## Why FlipGate?

Teams usually ship a quantised model once aggregate accuracy looks unchanged. But [Dutta et al. 2024](https://arxiv.org/abs/2407.09141) showed that aggregate accuracy can hide many per-question flips. FlipGate adds:

1. **Noise floor measurement** — bf16-vs-bf16 flip rate under greedy decoding (temp 0) across batch sizes 1/8/32, with/without `batch_invariant_ops`
2. **Statistical rigor** — A candidate only fails when right-to-wrong flips are significantly above the measured floor (p<0.05)
3. **Hallucination detection** — FedProc FAR/DFARS registry check (no LLM judge)

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install FlipGate
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run tests
pytest

# Check a candidate against baseline
flipgate check --baseline bf16 --candidate awq --dataset gsm8k

# Generate noise floor report
flipgate noise-floor --model base --engine vllm
```

## Project Structure

```
flipgate/
├── src/flipgate/
│   ├── scorers/          # Rule-based scorers (GSM8K, IFEval, FedProc)
│   ├── engines/          # Inference engines (vLLM, HF, llama.cpp)
│   ├── stats/            # Statistical tests (McNemar, bootstrap)
│   ├── manifest.py       # Frozen eval manifest loader
│   ├── store.py          # Per-item JSONL results store
│   └── cli.py            # CLI interface
├── tests/                # pytest tests
├── configs/
│   └── manifest.yaml     # Frozen eval manifest
└── data/
    ├── eval/             # Evaluation datasets
    └── results/          # Per-item results (JSONL)
```

## Milestones

1. **Harness + frozen manifest** (4d) — ✅ In progress
2. **Noise floor** (4d) — bf16-vs-bf16 flips at batch 1/8/32
3. **Quant/serving sweep** (5d) — AWQ, GPTQ-Int4, GGUF comparison
4. **The gate** (4d) — CLI + GitHub Action
5. **Publish** (3d) — HF dataset, README demo, dev.to article

## Hardware

- **GPU**: RTX 3060 12GB
- **Model scope**: Qwen/Qwen2.5-3B-Instruct (~6.5 GB bf16)
- **Budget**: ~25-35 GPU-hours for full sweep

## Target Roles

This project demonstrates:
- LLMOps regression gates and release engineering
- Inference optimisation trade-offs (AWQ, GPTQ, GGUF; vLLM vs HF vs llama.cpp)
- Statistical rigour in evaluation (paired tests, bootstrap CIs, noise floors)
- Reproducibility and determinism of LLM inference
- CI/CD for models (GitHub Actions, MLflow)

Aimed at: Noeon Research, PayPay Card, Money Forward, Treasure AI, Citadel AI (Tokyo/Japan ML Platform roles)

## License

MIT

## Article

Read the full story: [FlipGate: Building a Statistical Release Gate for Quantised LLMs](./devto_article.md)

The article covers:
- Why accuracy isn't enough for evaluating quantised models
- How FlipGate measures noise floors and detects per-item flips
- Results from evaluating Qwen2.5-3B (bf16 vs AWQ vs GPTQ-Int4)
- Challenges we hit (vLLM CUDA issues, statistical power limits)
- Why this matters for LLMOps teams

## Results Summary

### Large-Scale Evaluation (200 items)

| Model | Accuracy | Items | Time |
|-------|----------|-------|------|
| bf16 (baseline) | 34.0% | 200 | 25 min |

### Quantization Comparison (30 items)

| Model | Accuracy | Right→Wrong | Wrong→Right | McNemar p |
|-------|----------|-------------|-------------|-----------|
| bf16 (baseline) | 30.0% | — | — | — |
| AWQ (4-bit) | 26.7% | 3 | 2 | 1.0000 |
| GPTQ-Int4 | 33.3% | 0 | 1 | 1.0000 |

**Finding**: No statistically significant regressions detected (p = 1.0).

**Limitation**: 30 items isn't enough statistical power. Production use requires 200-500+ items.

**Known Issue**: Quantized models (AWQ, GPTQ) failed to run at scale due to Marlin kernel compilation issues with torch 2.13.0 / CUDA 13.0.

## Dataset

All evaluation results are published: https://huggingface.co/datasets/raihan-js/flipgate-results

- 21 evaluation runs
- 530 items evaluated
- Per-item prompts, responses, and scores

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

### Large-Scale Evaluation (1,000 items, HF generate)

| Model | Accuracy | Δ vs bf16 | Items |
|-------|----------|-----------|-------|
| bf16 (baseline) | 33.0% (330/1000) | — | 1000 |
| AWQ (4-bit) | 43.1% (431/1000) | **+10.1 pts** | 1000 |
| GPTQ-Int4 | 37.3% (373/1000) | **+4.3 pts** | 1000 |

**Flip analysis (1,000 common items):**

| Comparison | Right→Wrong | Wrong→Right | R→W rate | 95% CI | McNemar p |
|------------|-------------|-------------|----------|--------|-----------|
| bf16 vs AWQ | 77 | 178 | 7.7% | [6.3%, 9.5%] | **<0.0001** |
| bf16 vs GPTQ-Int4 | 89 | 132 | 8.9% | [7.2%, 10.7%] | **0.0047** |

### Engine vs Quantization Control (llama.cpp, 200 items)

| Model | Accuracy | Δ vs bf16 |
|-------|----------|-----------|
| f16 (llama.cpp) | 31.0% (62/200) | −2.0 pts |
| q4_K_M (llama.cpp) | 36.5% (73/200) | +3.5 pts |

| Comparison | Right→Wrong | Wrong→Right | R→W rate | 95% CI | McNemar p |
|------------|-------------|-------------|----------|--------|-----------|
| bf16 vs f16 (llama.cpp) | 24 | 18 | 12.0% | [8.0%, 16.5%] | 0.4404 |
| bf16 vs q4_K_M (llama.cpp) | 21 | 26 | 10.5% | [6.5%, 15.0%] | 0.5596 |

### Engine vs Quantization Control (llama.cpp f16 row, 200 items)

| Comparison | Isolates | Right→Wrong | Wrong→Right | R→W rate | 95% CI | McNemar p |
|------------|----------|-------------|-------------|----------|--------|-----------|
| bf16 (HF) vs f16 (llama.cpp) | engine only | 24 | 18 | 12.0% | [8.0%, 16.5%] | 0.440 |
| f16 vs q4_K_M (same engine) | quantization only | 14 | 25 | 7.0% | [3.5%, 10.5%] | 0.109 |
| bf16 (HF) vs q4_K_M (GGUF) | confounded total | 21 | 26 | 10.5% | [6.5%, 15.0%] | 0.560 |

### IFEval (541 prompts, rule-checked, independent reimplementation of the 25 published rules)

| Model | Accuracy | Δ vs bf16 |
|-------|----------|-----------|
| bf16 (baseline) | 59.5% (322/541) | — |
| AWQ (4-bit) | 56.7% (307/541) | −2.8 pts |
| GPTQ-Int4 | 58.6% (317/541) | −0.9 pts |

| Comparison | Right→Wrong | Wrong→Right | R→W rate | 95% CI | McNemar p |
|------------|-------------|-------------|----------|--------|-----------|
| bf16 vs AWQ | 56 | 41 | 10.4% | [7.8%, 12.9%] | 0.155 |
| bf16 vs GPTQ-Int4 | 45 | 40 | 8.3% | [6.1%, 10.5%] | 0.664 |

### FedProc Hallucination (155 real-FAR records, registry-checked)

| Model | No-hallucination rate | Δ vs bf16 | New hallucinations | McNemar p |
|-------|----------------------|-----------|-------------------|-----------|
| bf16 (baseline) | 81.9% (127/155) | — | — | — |
| AWQ (4-bit) | 67.1% (104/155) | −14.8 pts | 34 | 0.0010 |
| GPTQ-Int4 | 74.2% (115/155) | −7.7 pts | 20 | 0.0376 |

### BFCL Function Calling (550 tasks: 400 simple + 100 exec-simple + 50 exec-multiple)

| Model | Rate | Δ vs bf16 |
|-------|------|-----------|
| bf16 (baseline) | 71.8% (395/550) | — |
| AWQ (4-bit) | 74.0% (407/550) | +2.2 pts |
| GPTQ-Int4 | 71.1% (391/550) | −0.7 pts |

| Comparison | Right→Wrong | Wrong→Right | R→W rate | 95% CI | McNemar p |
|------------|-------------|-------------|----------|--------|-----------|
| bf16 vs AWQ | 6 | 18 | 1.1% | [0.4%, 2.0%] | **0.0247** |
| bf16 vs GPTQ-Int4 | 14 | 10 | 2.5% | [1.3%, 4.0%] | 0.5403 |

### Quantization Comparison (30 items)

| Model | Accuracy | Right→Wrong | Wrong→Right | McNemar p |
|-------|----------|-------------|-------------|-----------|
| bf16 (baseline) | 30.0% | — | — | — |
| AWQ (4-bit) | 26.7% | 3 | 2 | 1.0000 |
| GPTQ-Int4 | 33.3% | 0 | 1 | 1.0000 |

**Finding**: No statistically significant regressions detected (p = 1.0).

**Limitation**: 30 items isn't enough statistical power. Production use requires 200-500+ items.

**Known Issue**: Quantized models (AWQ, GPTQ) failed to run at scale due to Marlin kernel compilation issues with torch 2.13.0 / CUDA 13.0.

## Gate Demo: Catching a Broken Candidate

A serving config change truncated generation to 32 tokens, cutting off chain-of-thought reasoning:

```
$ flipgate check --baseline <bf16-run> --candidate <truncated-run> --dataset gsm8k

| Baseline accuracy       | 33.3%            |
| Candidate accuracy      | 0.0%             |
| Right-to-wrong flips    | 10               |
| Wrong-to-right flips    | 0                |
| McNemar p-value         | 0.0044           |

FAIL: significant right-to-wrong flip asymmetry
```

The gate fails the candidate and writes a Markdown report listing all 10 flipped items. Try it: `PYTHONPATH=src python -m flipgate.cli check --help`.

## Dataset

All evaluation results are published: https://huggingface.co/datasets/raihan-js/flipgate-results

- 21 evaluation runs
- 530 items evaluated
- Per-item prompts, responses, and scores

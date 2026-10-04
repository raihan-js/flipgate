# AGENTS.md — flipgate

Portfolio project for Raihan Sikder. Target roles: Noeon Research (Senior ML Engineer, LLMOps), PayPay Card, Money Forward, Treasure AI, Citadel AI.

## Project: FlipGate

A CLI + GitHub Action release gate for quantised/re-served LLMs. Counts per-item right-to-wrong answer flips vs. a measured bf16 noise floor, using paired statistics (McNemar, paired bootstrap) instead of aggregate accuracy.

## Why this project

Teams ship quantised models when aggregate accuracy looks unchanged. Dutta et al. 2024 showed aggregate accuracy hides per-question flips. FlipGate adds: (1) a measured noise floor first — under greedy decoding, seeds change nothing; batch size and kernel choice do; (2) a candidate fails only when right-to-wrong flips are significantly above the floor (p<0.05); (3) a hallucination column checking clause numbers against the 1,032-clause FAR/DFARS registry, no LLM judge.

## Key results

- GSM8K-1000: bf16 33.0%, AWQ 43.1% (p<0.0001), GPTQ 37.3% (p=0.0047). 77–89 correct answers broke silently behind accuracy gains.
- FedProc hallucination: AWQ p=0.0010, GPTQ p=0.0376 (both significant rises).
- BFCL-550: AWQ p=0.0247, GPTQ p=0.5403 (honest null).
- Noise floor: 0 flips HF generate batch 1/8; 0 flips vLLM batch 32/8; 1 W2R flip in 200 at vLLM batch 1.
- Engine control: llama.cpp f16 isolates engine (12%) from quantisation (7%) effects.

## Stack

Python, PyTorch, vLLM, llama.cpp (GGUF), HuggingFace Transformers, scipy, MLflow, GitHub Actions.

## Compute

One RTX 3060 12GB. Qwen2.5-3B-Instruct bf16/AWQ/GPTQ-Int4/GGUF. ~25-35 GPU-hours for the full sweep.

## Eval sets

- GSM8K (fixed 1,000-item slice, exact numeric match)
- IFEval (541 prompts, 25 rule-based checkers)
- FedProc real-FAR slice (155 records, registry hallucination)
- BFCL-550 (simple + exec gold, function calling)

## Conventions

- Python 3.10+, pytest for all scorers and stats helpers.
- Every eval run logs to MLflow (local `./mlruns`).
- Per-item results stored as JSONL, never overwritten — append with run metadata.
- All flip-rate claims cite exact run ID and dataset revision.
- No LLM-as-judge anywhere. Rule-based scorers only.

## Development

```bash
cd flipgate
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # tests + CLI + gate (light, no torch)
pip install -e ".[gpu]"        # adds torch/transformers/vllm for eval scripts
PYTHONPATH=src pytest tests/ -v  # 127 tests
PYTHONPATH=src python -m flipgate.cli --help
```

## Model files (re-downloadable, not in git)

Qwen2.5-3B-Instruct, -AWQ, -GPTQ-Int4, -GGUF, Qwen2.5-0.5B-Instruct from HuggingFace. `data/models/` was cleared to free disk; re-download to re-run evals.

## Current status — COMPLETE

- 127 tests, CI green, GitHub repo, HF dataset, dev.to article, social kit
- Repo: https://github.com/raihan-js/flipgate
- HF: https://huggingface.co/datasets/raihan-js/flipgate-results

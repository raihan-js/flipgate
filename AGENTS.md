# AGENTS.md — flipgate

Portfolio project for Raihan Sikder. Target roles: Noeon Research (Senior ML Engineer, LLMOps), PayPay Card, Money Forward, Treasure AI, Citadel AI.

## Project: FlipGate

A CLI + GitHub Action release gate for quantised/re-served LLMs. Counts per-item right-to-wrong answer flips vs. a measured bf16 noise floor, using paired statistics (McNemar, paired bootstrap) instead of aggregate accuracy.

## Why this project

Teams ship quantised models when aggregate accuracy looks unchanged. Dutta et al. 2024 showed aggregate accuracy hides per-question flips. FlipGate adds: (1) a measured noise floor first — under greedy decoding, seeds change nothing; batch size and kernel choice do; (2) a candidate fails only when right-to-wrong flips are significantly above the floor (p<0.05); (3) a hallucination column checking clause numbers against the 1,032-clause FAR/DFARS registry, no LLM judge.

## Key results (re-run 2026-10-05; the first GSM8K sweeps used a 256-token cap and are superseded)

- GSM8K-1000 (1,024-token cap, 0% truncated, batch size 1): bf16 79.7%, AWQ 76.2% (R→W 91, W→R 56, p=0.0050, FAIL), GPTQ 76.0% (91 / 54, p=0.0028, FAIL). The old 33.0 / 43.1 / 37.3% and "AWQ +10.1" are retired.
- Answer extractor flips the sign: strict v1 extractor gives AWQ +3.2 pts (p=0.050) because quantised models drop `\boxed{}` (57.7% bf16 vs 28.3% AWQ, 32.8% GPTQ); robust v2 gives −3.5.
- Noise floor: 0 flips for repeat runs at a fixed batch size (HF generate, eager vLLM). Changing batch size 1→8 (bf16, 200 items): 124/200 responses differ in text, 6 R→W + 7 W→R = 3.0% [1.0%, 5.5%] R→W floor. Text differs between batch 1 and 8 for 14/24 (bf16), 5/24 (AWQ), 6/24 (GPTQ) prompts.
- FedProc hallucination: AWQ p=0.0010, GPTQ p=0.0376 (both significant rises; unaffected by the cap).
- BFCL-550: AWQ p=0.0247, GPTQ p=0.5403 (unaffected).
- IFEval-541: AWQ −2.8 pts (p=0.155), GPTQ −0.9 (p=0.664).
- llama.cpp engine-control rows WITHDRAWN (256-token cap vs truncated baseline, different system prompt than HF). `scripts/run_gguf.py` now uses the HF chat template and the manifest cap, but the installed llama-cpp-python is CPU-only (est. 13 h for 200 items); planned replacement is HF vs vLLM on the same bf16 weights.

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
PYTHONPATH=src pytest tests/ -v  # 146 tests
PYTHONPATH=src python -m flipgate.cli --help
```

## Model files (re-downloadable, not in git)

Qwen2.5-3B-Instruct, -AWQ, -GPTQ-Int4, -GGUF, Qwen2.5-0.5B-Instruct from HuggingFace. `data/models/` was cleared to free disk; re-download to re-run evals.

## Current status — COMPLETE

- 146 tests, CI green, GitHub repo, HF dataset (rebuilt as flat tables), dev.to article (rewritten for the re-run), social kit
- Repo: https://github.com/raihan-js/flipgate
- HF: https://huggingface.co/datasets/raihan-js/flipgate-results

# AWQ Raised GSM8K Accuracy by 10 Points and Broke 77 Correct Answers

*FlipGate: a release gate that counts per-item answer flips against a measured noise floor, and what it found, including a bug in my own baseline.*

---

## The Problem

When you quantise an LLM from bf16 to INT4, the standard metric is accuracy: does the quantised model get the same percentage of questions right?

But accuracy hides something critical. A model can keep the same accuracy while flipping many individual answers from right to wrong, compensated by wrong-to-right flips elsewhere. For production systems, these per-item flips can break user trust, violate compliance requirements, or cause subtle bugs that aggregate metrics miss.

**FlipGate** is a release gate that counts per-item answer flips, compares them against a measured noise floor, and uses paired statistical tests to determine if a quantisation change actually regressed the model.

---

## How It Works

### 1. Measure the Noise Floor

Even with temperature=0 (greedy decoding), LLM inference isn't perfectly deterministic. Batch size, kernel choice, and floating-point reduction order all introduce tiny variations.

We measure this "noise floor" by running the same bf16 model multiple times and counting how often answers flip between runs. This establishes the baseline: how many flips are just noise?

```python
# Pseudocode
baseline_run_1 = evaluate(model, dataset)
baseline_run_2 = evaluate(model, dataset)
noise_floor = count_flips(baseline_run_1, baseline_run_2)
```

### 2. Evaluate the Candidate

Run the quantised model (AWQ, GPTQ, GGUF, etc.) on the same dataset with the same prompts.

### 3. Compare with Paired Statistics

For each item, we know:
- Did the baseline get it right?
- Did the candidate get it right?

This gives us four categories:
- **Both correct**: No flip
- **Both wrong**: No flip
- **Right → Wrong**: Regression (bad)
- **Wrong → Right**: Improvement (good)

We use **McNemar's test** to determine if the right→wrong flips are statistically significant compared to wrong→right flips. We also compute **bootstrap confidence intervals** for the flip rate.

### 4. Gate Decision

If the candidate's flip rate exceeds the noise floor by a configurable margin (and the difference is statistically significant at p < 0.05), the gate **fails**. Otherwise, it **passes**.

---

## What We Built

### Architecture

```
flipgate/
├── src/flipgate/
│   ├── scorers/          # GSM8K, IFEval, FedProc (registry hallucination)
│   ├── engines/          # HF generate, vLLM, llama.cpp
│   ├── stats/            # McNemar's test, paired bootstrap CI
│   ├── store.py          # Per-item JSONL results (append-only)
│   ├── manifest.py       # Frozen eval config (pinned versions)
│   └── cli.py            # flipgate check, flipgate info, etc.
├── tests/                # 127 pytest tests
├── configs/manifest.yaml # Pinned models, datasets, sampling params
└── .github/actions/      # GitHub Action for CI/CD
```

### Key Design Decisions

**1. Frozen Manifest**

Every eval run records:
- Exact model versions (git SHAs)
- Dataset revisions
- Sampling parameters (temperature, top_p, etc.)
- Engine versions
- Manifest SHA

This ensures reproducibility. If someone questions your results, you can point to the exact configuration.

**2. Append-Only Results Store**

Results are stored as JSONL files, one per item. We never overwrite—only append. This means:
- Full audit trail
- Easy to re-analyze with different scorers
- Simple to diff between runs

**3. No LLM-as-Judge**

All scorers are rule-based:
- **GSM8K**: Exact numeric match on final answer
- **IFEval**: Instruction following checks (word count, formatting, etc.)
- **FedProc**: Registry hallucination check (does this clause number exist?)

LLM judges are slow, expensive, and non-deterministic. Rule-based scorers are fast, cheap, and reproducible.

---

## Results

We evaluated **Qwen2.5-3B-Instruct** in five configurations on **GSM8K** (math reasoning):

### Large-Scale Evaluation (1,000 items, HF generate)

| Model | Accuracy | Δ vs bf16 | Items |
|-------|----------|-----------|-------|
| bf16 (baseline) | 33.0% (330/1000) | — | 1000 |
| AWQ (4-bit) | 43.1% (431/1000) | **+10.1 pts** | 1000 |
| GPTQ-Int4 | 37.3% (373/1000) | **+4.3 pts** | 1000 |

f16 and q4_K_M (llama.cpp, 200 items each) are covered in the engine-control section below.

**This is the FlipGate story in one table, with a harness caveat in the next section.** Both quantized models *improved* accuracy — yet both broke previously-correct answers, and at n=1,000 **both differences are statistically significant**:

| Comparison | Right→Wrong | Wrong→Right | R→W rate | 95% CI | McNemar p |
|------------|-------------|-------------|----------|--------|-----------|
| bf16 vs AWQ | 77 | 178 | 7.7% | [6.3%, 9.5%] | **<0.0001** |
| bf16 vs GPTQ-Int4 | 89 | 132 | 8.9% | [7.2%, 10.7%] | **0.0047** |

77–89 correct answers broke silently behind accuracy gains. A team shipping on accuracy alone would never see this.

### Engine vs Quantization: The Control Row That Earned Its Place

GGUF runs on llama.cpp, not on HF generate — so engine and quantization are confounded in a naive comparison. The spec requires a llama.cpp f16 control row precisely for this, and the data vindicates it:

| Comparison | What it isolates | Right→Wrong | Wrong→Right | R→W rate | 95% CI | McNemar p |
|------------|-----------------|-------------|-------------|----------|--------|-----------|
| bf16 (HF) vs f16 (llama.cpp) | **engine only** | 24 | 18 | 12.0% | [8.0%, 16.5%] | 0.440 |
| f16 vs q4_K_M (same engine) | **quantization only** | 14 | 25 | 7.0% | [3.5%, 10.5%] | 0.109 |
| bf16 (HF) vs q4_K_M | confounded total | 21 | 26 | 10.5% | [6.5%, 15.0%] | 0.560 |

Just swapping the inference engine — same weights, same precision — flipped 42 of 200 items (21%). The engine effect (12.0%) is *larger* than the pure quantization effect (7.0%). Without the f16 control, we would have blamed quantization for flips the engine caused. Neither difference is statistically significant at p < 0.05, but the decomposition itself is the point: measure the control, or your attribution is guesswork.

Against our measured noise floor of 0% (bf16-vs-bf16, HF generate, temp 0, batch 1/8), every one of these flips exceeds the floor.

> **A note on the floor:** the original design expressed flips as a *multiple* of the noise floor (e.g. "4x the floor"). With a measured floor of exactly 0%, that ratio is undefined — so we report flips as an absolute rate above the floor with its 95% CI instead. The gate logic is unchanged: any flip rate above floor + margin trips it.

### IFEval: Instruction Following Degrades (541 prompts)

GSM8K showed accuracy going up. IFEval — 541 rule-checked instruction-following prompts scored with our own reimplementation of the 25 published IFEval rules — shows the other side:

| Model | Accuracy | Δ vs bf16 |
|-------|----------|-----------|
| bf16 (baseline) | 59.5% (322/541) | — |
| AWQ (4-bit) | 56.7% (307/541) | **−2.8 pts** |
| GPTQ-Int4 | 58.6% (317/541) | **−0.9 pts** |

| Comparison | Right→Wrong | Wrong→Right | R→W rate | 95% CI | McNemar p |
|------------|-------------|-------------|----------|--------|-----------|
| bf16 vs AWQ | 56 | 41 | 10.4% | [7.8%, 12.9%] | 0.155 |
| bf16 vs GPTQ-Int4 | 45 | 40 | 8.3% | [6.1%, 10.5%] | 0.664 |

Here the aggregate *does* move — AWQ loses nearly 3 points — but the flip counts tell the fuller story: 56 and 45 previously-passing instructions broke. The two task families together make the case no single number can: on math reasoning the quantized models looked *better* while breaking answers; on instruction following they look *worse*, and the flips quantify exactly how much worse per item.

### FedProc: Hallucination Rises (155 real-FAR records)

The third column needs no accuracy at all — just a registry. We prompt each model with a clause topic and check every cited FAR/DFARS number against the 1,128-entry registry built from ECFR Title 48. Score 1.0 means no fabricated numbers; 0.0 means at least one hallucinated clause. Only the real-FAR slice is used; the 65 Claude-written synthetic records stay out.

| Model | No-hallucination rate | Δ vs bf16 | New hallucinations | Fixed | McNemar p |
|-------|----------------------|-----------|-------------------|-------|-----------|
| bf16 (baseline) | 81.9% (127/155) | — | — | — | — |
| AWQ (4-bit) | 67.1% (104/155) | **−14.8 pts** | 34 | 11 | **0.0010** |
| GPTQ-Int4 | 74.2% (115/155) | **−7.7 pts** | 20 | 8 | **0.0376** |

Both increases are statistically significant. This is the gate's second tripwire firing exactly as designed: `fail when registry hallucination rises`. Quantization doesn't just flip answers — it fabricates clause numbers, and the registry check catches it with no LLM judge involved.

### BFCL: Function Calling Is the Most Stable Capability (550 tasks)

Three BFCL slices — simple (400), exec-simple (100, exact gold match), exec-multiple (50) — scored as executable-call rate:

| Model | Executable-call rate | Δ vs bf16 |
|-------|---------------------|-----------|
| bf16 (baseline) | 71.8% (395/550) | — |
| AWQ (4-bit) | 74.0% (407/550) | **+2.2 pts** |
| GPTQ-Int4 | 71.1% (391/550) | −0.7 pts |

| Comparison | Right→Wrong | Wrong→Right | R→W rate | 95% CI | McNemar p |
|------------|-------------|-------------|----------|--------|-----------|
| bf16 vs AWQ | 6 | 18 | 1.1% | [0.4%, 2.0%] | **0.0247** |
| bf16 vs GPTQ-Int4 | 14 | 10 | 2.5% | [1.3%, 4.0%] | 0.5403 |

At n=400 AWQ sat at p=0.055 (borderline); at n=550 it crossed into significance. Function calling remains the most stable family under quantization — structured output with explicit schemas resists flips better than free-form reasoning.

### Minimum Detectable Effect

How many items do you need to catch a regression? For paired binary outcomes (McNemar, 80% power, α = 0.05, baseline accuracy 34%):

| Accuracy drop | Items needed |
|---------------|--------------|
| 1 pt | ~3,900 |
| 2 pt | ~1,175 |
| 5 pt | ~280 |

The 1,000-item runs sit between the 2-point and 5-point rows; the 200-item llama.cpp rows can only detect ~5-point drops. Detecting a 1-point drop needs thousands of items, which is why per-item flip tracking matters more than waiting for aggregate accuracy to move.

### Noise Floor: HF generate and vLLM

Before counting flips you need to know how many happen with nothing changed. With temperature 0 and batch sizes 1 and 8, HF generate gave **0 flips** between repeated bf16 runs. vLLM 0.30.0 (eager mode, `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`) gave **0 flips at batch 32 and batch 8** over 1,000 items each, and 1 flip in 200 items at batch 1 (wrong→right). On this stack the noise floor is effectively zero, so every flip counted above is a real difference between models or engines. Other stacks (CUDA graphs, other kernels, multi-GPU) can differ, which is why you measure it.

### A Bug in My Own Baseline: the 256-Token Cap

The GSM8K script generates with `max_new_tokens=256` (the manifest says 2048; the manifest was not what this script used). Qwen2.5-3B-Instruct often needs more than 256 tokens for chain-of-thought, so most responses are cut off before a final answer. Measured on the three 1,000-item runs:

| Run | Responses with a `\boxed{}` final answer | Accuracy among those |
|---|---|---|
| bf16 | 16.3% | 91.4% |
| AWQ | 8.0% | 88.8% |
| GPTQ-Int4 | 8.6% | 79.1% |

So the 33-43% above mostly measures "finished within 256 tokens and the extractor found the answer", not reasoning quality, and absolute accuracy should not be compared with published GSM8K numbers. The flip analysis is still a valid comparison under *this* harness, but part of the accuracy swing is verbosity.

**Scorer sensitivity.** Re-scoring the same stored responses with a more robust extractor (last `\boxed{}`, then `####`, then "answer is", then the last number with thousands separators handled) gives bf16 43.0%, AWQ 50.2%, GPTQ 45.0%, and:

| Comparison (robust extractor) | Right→Wrong | Wrong→Right | Net | McNemar p |
|---|---|---|---|---|
| bf16 vs AWQ | 69 | 141 | +7.2 pts | 7.6e-7 |
| bf16 vs GPTQ-Int4 | 78 | 98 | +2.0 pts | 0.15 |

The AWQ result survives; the GPTQ GSM8K result does not. A re-run with a 1,024-token cap is the right fix and is planned. The FedProc registry check (short outputs) and BFCL (short function calls) are not affected by this cap.

---

## Challenges & Lessons Learned

### 1. Read the raw responses

The most expensive mistake in this project was not a kernel or a statistic. It was trusting an accuracy number without reading the outputs: the 256-token cap truncated most GSM8K answers, and it only became visible when I read the failing responses and saw them stop mid-equation. A frozen manifest that does not match what the script actually ran is worse than no manifest. FlipGate's own demo (a 32-token cap that the gate fails correctly) describes the same failure in a milder form.

### 2. Sample size decided every conclusion

At n=30 nothing was detectable (p=1.0 everywhere). BFCL for AWQ sat at p=0.055 at n=400 and crossed to p=0.0247 at n=550. Two of the GSM8K conclusions change with the answer extractor. Treat a p-value as a statement about one harness at one sample size.

### 3. The noise floor was zero, and that is a result

I expected run-to-run nondeterminism and measured none on HF generate (batch 1 and 8) or on vLLM in eager mode (batch 32 and 8). That makes every flip attributable, and it is specific to this stack.

### 4. Tooling friction

vLLM's flashinfer sampler needed a newer CUDA toolchain than I had (`nvcc fatal: Unknown option '--compress-mode=size'`); the fix was disabling it and forcing FlashAttention. gptqmodel's Marlin kernels did not compile against torch 2.13 / CUDA 13, so the quantised runs use HF generate. Test your inference engine on the target hardware before committing to an evaluation pipeline.

---

## Why This Matters

### For LLMOps Teams

1. **Catch silent regressions**: Accuracy can stay flat while individual answers flip. FlipGate catches this.

2. **Reproducible evaluations**: Frozen manifests and append-only results mean you can audit any decision months later.

3. **Statistical rigor**: McNemar's test and bootstrap CIs give you confidence intervals, not just point estimates.

4. **CI/CD integration**: The GitHub Action lets you gate model deployments automatically.

### For the Community

We're releasing FlipGate as open source because:
- The problem is real (accuracy hides flips)
- The solution is straightforward (paired tests + noise floor)
- The tooling is missing (most teams just check accuracy)

---

## What's Next

1. **Re-run GSM8K with a 1,024-token cap** and report both harnesses side by side. This is the fix for the biggest weakness above.
2. **Larger models (7B, 13B)** on rented GPUs, and a Marlin build so quantised models can run on vLLM.
3. **More task families**, for example code generation, with execution-based scoring.
4. **Noise floor under non-eager vLLM** and other stacks.

---

## Try It Yourself

```bash
# Clone the repo
git clone https://github.com/raihan-js/flipgate
cd flipgate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Check a candidate against baseline
flipgate check \
  --baseline <baseline_run_id> \
  --candidate <candidate_run_id> \
  --dataset gsm8k
```

**Repo**: https://github.com/raihan-js/flipgate  
**Dataset**: https://huggingface.co/datasets/raihan-js/flipgate-results

---

## Limitations

- **GSM8K harness**: 256-token generation cap and a strict answer extractor (see above). Absolute accuracy is not comparable with published numbers; the GPTQ GSM8K result is not robust to the extractor.
- **One model**: Qwen2.5-3B-Instruct only. Results may not generalise to other families or sizes.
- **Quantised runs on HF generate only**: Marlin kernels did not compile here, so quantised models were not served through vLLM.
- **Underpowered controls**: the llama.cpp rows are 200 items and cannot detect small effects.
- **Noise floor is for one stack**: measured on HF generate and eager vLLM at temperature 0.

---

## Conclusion

Aggregate accuracy hides item-level change. On GSM8K both quantised variants gained accuracy while 69-89 previously correct answers broke (the AWQ result survives a stricter re-score; GPTQ's does not). On the FAR registry check both fabricated more clauses (AWQ added 34, McNemar p=0.001). On function calling almost nothing moved. Which family you gate on matters, and so does reading the outputs.

The tool is a CLI and a GitHub Action; the framework is extensible with your own scorers, engines and datasets.

---

*Questions or corrections? Open an issue on GitHub: github.com/raihan-js/flipgate*

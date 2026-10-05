![FlipGate GSM8K flip counts](https://raw.githubusercontent.com/raihan-js/flipgate/HEAD/images/flipgate-results.png)

# AWQ Looked 10 Points Better on GSM8K Until I Stopped Truncating the Answers

*FlipGate: a release gate that counts per-item answer flips against a measured noise floor, what it found on Qwen2.5-3B, and the bug in my own baseline that I had to fix first.*

---

## The Problem

When you quantise an LLM from bf16 to INT4, the standard metric is accuracy: does the quantised model get the same percentage of questions right?

But accuracy hides something critical. A model can keep the same accuracy while flipping many individual answers from right to wrong, compensated by wrong-to-right flips elsewhere. For production systems, these per-item flips can break user trust, violate compliance requirements, or cause subtle bugs that aggregate metrics miss.

**FlipGate** is a release gate that counts per-item answer flips, compares them against a measured noise floor, and uses paired statistical tests to determine if a quantisation change actually regressed the model.

---

## How It Works

### 1. Measure the Noise Floor

Even with temperature=0 (greedy decoding), the same model can answer differently when the serving setup changes. Batch size, kernel choice, and floating-point reduction order all move the numbers.

We measure this "noise floor" by running the same bf16 model under the conditions that vary in practice and counting how often *correctness* flips. This establishes the baseline: how many flips are just noise?

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
├── tests/                # 146 pytest tests
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

Qwen2.5-3B-Instruct on four task families. All sampling is greedy; the GSM8K sweeps use a 1,024-token cap and batch size 1.

### GSM8K: both quantised models fail the gate (1,000 items)

| Model | Accuracy | Δ vs bf16 | Items |
|-------|----------|-----------|-------|
| bf16 (baseline) | 79.7% (797/1000) | — | 1000 |
| AWQ (4-bit) | 76.2% (762/1000) | **−3.5 pts** | 1000 |
| GPTQ-Int4 | 76.0% (760/1000) | **−3.7 pts** | 1000 |

| Comparison | Right→Wrong | Wrong→Right | R→W rate | 95% CI | McNemar p | Gate |
|------------|-------------|-------------|----------|--------|-----------|------|
| bf16 vs AWQ | 91 | 56 | 9.1% | [7.3%, 10.9%] | **0.0050** | FAIL |
| bf16 vs GPTQ-Int4 | 91 | 54 | 9.1% | [7.4%, 10.9%] | **0.0028** | FAIL |

Each quantised model broke 91 answers that bf16 got right, 11% of the 797, while fixing 54-56 that bf16 got wrong. Every response ended naturally (3,000 of 3,000, mean 274-297 tokens), so none of this is truncation.

### The noise floor: 0 for reruns, about 3% when the batch size changes

Repeating a run at a fixed batch size gives identical output on HF generate and on eager vLLM (0 flips). Changing the batch size does not. bf16 at batch size 1 vs batch size 8, same weights and prompts, first 200 items: **124 of 200 responses differed in text, but only 13 changed correctness** (6 right→wrong, 7 wrong→right). That is a right→wrong floor of **3.0% [1.0%, 5.5%]**, McNemar p = 1.0. The quantised models' 9.1% is three times that, and the floor's upper bound (5.5%) sits below the quantised models' lower bound (7.3%). A 24-prompt text check shows the same sensitivity for all three models: 14/24 (bf16), 5/24 (AWQ), 6/24 (GPTQ) responses differ between batch size 1 and 8. So baseline and candidate must use the same batch size, and the sweeps here use batch size 1.

> **A note on the floor:** the original design expressed flips as a *multiple* of the noise floor. With a floor of exactly 0% that ratio is undefined, so the gate reports the flip rate with its 95% CI and compares it with the measured floor plus a margin.

### The answer extractor decides the sign of the accuracy change

The quantised models drift away from the `\boxed{}` final-answer format: 57.7% of bf16 responses contain `\boxed`, against 28.3% for AWQ and 32.8% for GPTQ. The original strict extractor scores the *same stored responses* as bf16 60.0%, AWQ 63.2%, GPTQ 61.0%: AWQ apparently *better* by 3.2 points (109 right→wrong, 141 wrong→right, p = 0.050) and GPTQ unchanged (125 vs 135, p = 0.58). The robust extractor (last `\boxed{}`, then `####`, then "answer is", then the last number) gives the table above. A change in answer *format* is a real change a gate should surface, but it is not a change in reasoning, and a strict extractor will happily report it as an accuracy gain.

### IFEval: instruction following degrades (541 prompts)

541 rule-checked instruction-following prompts scored with my own reimplementation of the 25 published IFEval rules:

| Model | Accuracy | Δ vs bf16 |
|-------|----------|-----------|
| bf16 (baseline) | 59.5% (322/541) | — |
| AWQ (4-bit) | 56.7% (307/541) | **−2.8 pts** |
| GPTQ-Int4 | 58.6% (317/541) | **−0.9 pts** |

| Comparison | Right→Wrong | Wrong→Right | R→W rate | 95% CI | McNemar p |
|------------|-------------|-------------|----------|--------|-----------|
| bf16 vs AWQ | 56 | 41 | 10.4% | [7.8%, 12.9%] | 0.155 |
| bf16 vs GPTQ-Int4 | 45 | 40 | 8.3% | [6.1%, 10.5%] | 0.664 |

The aggregate moves a little and the flips quantify it: 56 and 45 previously passing instructions broke, though neither difference is significant at this sample size.

### FedProc: hallucination rises (155 real-FAR records)

The third family needs no accuracy at all, just a registry. Each model is prompted with a clause topic and every cited FAR/DFARS number is checked against the 1,128-entry registry built from eCFR Title 48. Score 1.0 means no fabricated numbers. Only the real-FAR slice is used; the 65 Claude-written synthetic records stay out.

| Model | No-hallucination rate | Δ vs bf16 | New hallucinations | Fixed | McNemar p |
|-------|----------------------|-----------|-------------------|-------|-----------|
| bf16 (baseline) | 81.9% (127/155) | — | — | — | — |
| AWQ (4-bit) | 67.1% (104/155) | **−14.8 pts** | 34 | 11 | **0.0010** |
| GPTQ-Int4 | 74.2% (115/155) | **−7.7 pts** | 20 | 8 | **0.0376** |

Both increases are significant: the gate's second tripwire, "fail when registry hallucination rises", fires with no LLM judge involved. (The outputs are short, so the generation cap does not matter here.)

### BFCL: function calling is the most stable family (550 tasks)

Three BFCL slices, simple (400), exec-simple (100, exact gold match) and exec-multiple (50), scored as executable-call rate:

| Model | Executable-call rate | Δ vs bf16 |
|-------|---------------------|-----------|
| bf16 (baseline) | 71.8% (395/550) | — |
| AWQ (4-bit) | 74.0% (407/550) | **+2.2 pts** |
| GPTQ-Int4 | 71.1% (391/550) | −0.7 pts |

| Comparison | Right→Wrong | Wrong→Right | R→W rate | 95% CI | McNemar p |
|------------|-------------|-------------|----------|--------|-----------|
| bf16 vs AWQ | 6 | 18 | 1.1% | [0.4%, 2.0%] | **0.0247** |
| bf16 vs GPTQ-Int4 | 14 | 10 | 2.5% | [1.3%, 4.0%] | 0.5403 |

Structured output with explicit schemas resists flips better than free-form reasoning. At n=400 AWQ sat at p=0.055; at n=550 it crossed to significance, so treat a p-value as a statement about one harness at one sample size.

### How many items do you need?

For paired binary outcomes (McNemar, 80% power, α = 0.05) the answer depends on how many items flip in either direction. In these runs about 14.7% of GSM8K items flipped (right→wrong or wrong→right) between bf16 and AWQ. Under that assumption, detecting a net change of 1 point needs about 11,500 items, 2 points about 2,900, 3.5 points about 940, and 5 points about 460. The 1,000-item sweeps sit right at the 3.5-point row, which is why the −3.5 result is significant but not by a wide margin, and why per-item flip counts are more informative than waiting for aggregate accuracy to move.

### A bug in my own baseline: the 256-token cap

The first version of this article was titled "AWQ raised GSM8K accuracy by 10 points and broke 77 correct answers". The GSM8K script generated with `max_new_tokens=256` (the manifest said 2048; the script never read it). Qwen2.5-3B-Instruct usually needs more than 256 tokens for chain-of-thought, so most responses were cut off before a final answer: only 16.3% (bf16), 8.0% (AWQ) and 8.6% (GPTQ) contained a `\boxed{}` answer, and the "accuracy" of 33.0% / 43.1% / 37.3% mostly measured who finished within 256 tokens. The more verbose bf16 model lost to the terser quantised ones for reasons unrelated to quality. I found it by reading failing responses that stopped mid-equation.

Two fixes followed. First, I re-ran everything with a 1,024-token cap (the table at the top of this section). The headline reversed: AWQ went from +10.1 points to −3.5, and GPTQ from +4.3 to −3.7. Second, FlipGate now refuses to compare runs like the old ones: each stored item carries a `finish_reason`, and `flipgate check` exits with **INVALID** if more than 10% of responses hit the cap in either run, or if the two runs' truncation rates differ by more than 5 points. The gate is supposed to catch exactly this kind of silent change, so it should not be fooled by one in its own harness.

I also withdrew an engine-versus-quantisation control (llama.cpp f16 against HF generate) that I had reported: it inherited the truncated baseline, and its llama.cpp prompt used a different system message from the HF chat template, so engine and prompt were confounded. Re-running it needs a GPU build of llama.cpp; the planned replacement is an HF-versus-vLLM comparison of the same bf16 weights.

---

## Challenges & Lessons Learned

### 1. Read the raw responses

The most expensive mistake in this project was not a kernel or a statistic. It was trusting an accuracy number without reading the outputs. A frozen manifest that does not match what the script actually ran is worse than no manifest. Reading the responses is also how I found the extractor effect: the quantised models change the *format* of their answers, not only the content.

### 2. A noise floor of zero was a statement about one condition

I measured 0 flips for repeated runs and nearly wrote "the floor is zero". It is zero when nothing changes; change the batch size and 62% of responses differ in text and 3% of items flip from right to wrong. Measure the floor under the changes you will actually make.

### 3. Sample size and extractor decided every conclusion

At n=30 nothing was detectable (p=1.0 everywhere). BFCL for AWQ crossed from p=0.055 to p=0.0247 between n=400 and n=550. The GSM8K sign flipped with the answer extractor. Treat a p-value as a statement about one harness at one sample size.

### 4. Tooling friction

vLLM's flashinfer sampler needed a newer CUDA toolchain than I had (`nvcc fatal: Unknown option '--compress-mode=size'`); the fix was disabling it and forcing FlashAttention. gptqmodel's Marlin kernels did not compile against torch 2.13 / CUDA 13, so the quantised runs use HF generate, and the installed llama.cpp binding is CPU-only. Test your inference engine on the target hardware before committing to an evaluation pipeline.

---

## Why This Matters

### For LLMOps teams

1. **Catch silent regressions**: here a 3.5-point accuracy drop came with 91 individually broken answers, and on the registry check 34 newly fabricated clauses.
2. **Know your floor**: a flip rate means little until you have measured what your own serving changes do with the weights held fixed.
3. **Reproducible evaluations**: frozen manifests and append-only results let you audit a decision, and let you find a bug in the harness months later.
4. **CI/CD integration**: the GitHub Action gates deployments automatically, and the harness-health checks make it fail loudly on a broken comparison.

### For the community

The problem is real (accuracy hides flips), the method is straightforward (paired tests plus a measured floor), and the tooling is missing: most teams only check accuracy.

---

## What's Next

1. **HF-versus-vLLM engine control** on the same bf16 weights, replacing the withdrawn llama.cpp rows.
2. **Larger models (7B, 13B)** on rented GPUs, and a Marlin build so quantised models can run on vLLM.
3. **More task families**, for example code generation with execution-based scoring, and a Japanese task.
4. **Noise floor under non-eager vLLM** and other stacks.

---

## Try It Yourself

```bash
git clone https://github.com/raihan-js/flipgate
cd flipgate
pip install -e ".[dev]"
pytest tests/ -v

# Check a candidate against a baseline (same batch size!)
flipgate check \
  --baseline <baseline_run_id> \
  --candidate <candidate_run_id> \
  --dataset gsm8k
```

**Repo**: https://github.com/raihan-js/flipgate  
**Dataset**: https://huggingface.co/datasets/raihan-js/flipgate-results

---

## Limitations

- **One model**: Qwen2.5-3B-Instruct only. Results may not generalise to other families or sizes.
- **Answer extractors matter**: the GSM8K accuracy change flips sign between a strict and a robust extractor; the robust one is used, and the format drift is itself a finding.
- **Quantised runs on HF generate only**: Marlin kernels did not compile here, so quantised models were not served through vLLM, and there is currently no engine-versus-quantisation control.
- **Noise floor**: the batch-size floor (3.0% [1.0%, 5.5%]) is from 200 GSM8K items on one stack; the floor under other stacks, kernels and multi-GPU setups can differ.
- **Cap and batch size**: the GSM8K results are for a 1,024-token cap at batch size 1. The IFEval (1,024-token cap), FedProc (128) and BFCL (256) runs predate the finish-reason field, so truncation there was not measured; their outputs are short, but that is an assumption.

---

## Conclusion

Aggregate accuracy hides item-level change, and a gate has to be right about its own measurements first. On GSM8K both quantised variants lost 3.5-3.7 points and broke 91 previously correct answers each, three times the batch-size noise floor, and the gate fails both. On the FAR registry check they fabricated more clauses (AWQ added 34, McNemar p=0.001); on function calling almost nothing moved. Which family you gate on matters, so does reading the outputs, and so does admitting when the first headline was a measurement artefact.

The tool is a CLI and a GitHub Action; the framework is extensible with your own scorers, engines and datasets.

---

*Questions or corrections? Open an issue on GitHub: github.com/raihan-js/flipgate*

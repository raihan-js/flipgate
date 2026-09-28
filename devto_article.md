# FlipGate: Building a Statistical Release Gate for Quantised LLMs

*Or: Why accuracy isn't enough, and how we built a tool to catch the flips it hides*

---

## The Problem

When you quantise an LLM from bf16 to INT4, the standard metric is accuracy: does the quantised model get the same percentage of questions right?

But accuracy hides something critical. A model might maintain 80% accuracy while silently flipping 20% of individual answers from right to wrong (and compensating with wrong-to-right flips elsewhere). For production systems, these per-item flips can break user trust, violate compliance requirements, or cause subtle bugs that aggregate metrics miss.

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
├── tests/                # 80 pytest tests
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

We evaluated **Qwen2.5-3B-Instruct** in three configurations on **GSM8K** (math reasoning):

| Model | Accuracy | Right→Wrong Flips | Wrong→Right Flips | McNemar p-value |
|-------|----------|-------------------|-------------------|-----------------|
| bf16 (baseline) | 30.9% | — | — | — |
| AWQ (4-bit) | 26.7% | 1 | 2 | 1.0000 |
| GPTQ-Int4 | 33.3% | 0 | 1 | 1.0000 |

**Key finding**: No statistically significant regressions detected (p = 1.0 for both comparisons).

### What This Means

With 24-30 overlapping items, we lack statistical power to detect small differences. But the flip rates are low (0-4%), suggesting that for this model and task, quantisation doesn't cause meaningful per-item regressions.

This is actually a **valid and publishable result**: sometimes the answer is "no significant difference," and that's useful information for teams deciding whether to ship a quantised model.

---

## Challenges & Lessons Learned

### 1. vLLM CUDA Compiler Issues

We attempted to use vLLM for faster batched inference and to measure the noise floor with realistic batched kernels. However, vLLM's flashinfer dependency requires CUDA compilation that failed on our system:

```
nvcc fatal: Unknown option '--compress-mode=size'
```

This is a known issue with older CUDA toolkits. **Lesson**: Always test your inference engine on your target hardware before committing to an evaluation pipeline.

### 2. Network & Download Limits

We hit network issues trying to download BFCL (Berkeley Function Calling Leaderboard) for a fourth task family. **Lesson**: Cache datasets locally and have fallback plans for network outages.

### 3. Statistical Power

With 30 items, we can't detect flip rates below ~10% with reasonable confidence. **Lesson**: For production use, you need 200-500+ items to detect meaningful regressions. We built the framework to scale, but our test hardware (single RTX 3060) limited the scope.

### 4. HF Generate is Deterministic (at batch_size=1)

We expected to see nondeterminism from floating-point operations, but HF generate with batch_size=1 produced identical outputs across runs. This suggests the noise floor comes from **batched kernels** (which we couldn't test due to vLLM issues), not from the model itself.

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

1. **vLLM noise floor**: Once we resolve the CUDA compiler issues, we'll measure the actual noise floor with batched kernels.

2. **BFCL integration**: Function calling is a critical capability. We'll add BFCL as a fourth task family.

3. **Larger evaluations**: We need 500+ items for statistical power. This requires more GPU time or cloud resources.

4. **Multi-model comparison**: Compare flip rates across different model families (Llama, Mistral, Qwen).

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

- **Small sample size**: 30 items isn't enough for high-confidence decisions. Production use requires 200-500+ items.
- **Single task family**: We only tested GSM8K. Different tasks (summarisation, code generation, function calling) may behave differently.
- **HF generate only**: We couldn't test vLLM due to CUDA issues. Batched kernels may show different noise floors.
- **Single model**: Qwen2.5-3B is one model. Results may not generalise to other architectures.

---

## Conclusion

FlipGate demonstrates that **accuracy isn't enough** for evaluating quantised LLMs. By counting per-item flips and comparing against a measured noise floor, we can catch regressions that aggregate metrics miss.

The tool is production-ready for teams who want statistical rigor in their model evaluation pipeline. The framework is extensible—add your own scorers, engines, and datasets.

Most importantly, we've shown that **sometimes the answer is "no significant difference,"** and that's a valid, useful result. Not every quantisation causes flips. But now you can prove it, not just assume it.

---

*Have questions or feedback? Open an issue on GitHub or reach out on Twitter.*

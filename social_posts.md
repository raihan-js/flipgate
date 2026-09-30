# FlipGate — Social Media Posts

## LinkedIn Post

**Quantization made my model smarter AND more hallucinating at the same time. Here's the data.**

I just finished FlipGate, an open-source release gate for quantized LLMs that counts per-item answer flips instead of trusting aggregate accuracy.

The setup: Qwen2.5-3B-Instruct (bf16 vs AWQ vs GPTQ-Int4), evaluated on GSM8K (200 items), IFEval (541 prompts), and FedProc FAR/DFARS registry checks (155 records) — all on a single RTX 3060.

What aggregate accuracy says:
- GSM8K: AWQ +11 pts, GPTQ +5 pts ("ship it!")
- IFEval: AWQ −2.8 pts, GPTQ −0.9 pts ("maybe not")

What per-item flips reveal:
- GSM8K: AWQ broke 13 previously-correct answers (McNemar p=0.0024); GPTQ broke 14
- IFEval: AWQ broke 56 instructions (10.4% flip rate); GPTQ broke 45
- FedProc: AWQ fabricated 34 new clause numbers (p=0.001); GPTQ fabricated 20 (p=0.038)

And the finding that surprised me most: just swapping the inference engine (HF generate → llama.cpp, same weights, same precision) flipped 42/200 items — a bigger effect than quantizing to 4-bit. Without an f16 engine-control row, I would have blamed quantization for flips the engine caused.

Key lessons:
1. Accuracy hides flips. Paired stats (McNemar + bootstrap CIs) don't.
2. Always measure a noise floor first. Mine was 0% — which made every flip meaningful.
3. Always run an engine control. Serving stack ≠ model.
4. Report your nulls. Half my comparisons weren't significant, and that's data too.

🔗 Repo: github.com/raihan-js/flipgate
📊 Data: huggingface.co/datasets/raihan-js/flipgate-results
📝 Full write-up with limitations: [dev.to link]

Built for ML Platform / LLMOps roles. Open to feedback — especially on the statistical methodology.

#MachineLearning #LLMOps #MLOps #Quantization #LLM #OpenSource #AIQuality

---
Images to attach (in order):
1. flipgate_thumbnail.png (cover)
2. accuracy_vs_flips.png
3. flip_directions.png
4. hallucination_detection.png
---

## X / Twitter Thread

**1/** Quantization made my model +11 points smarter and broke 13 correct answers at the same time.

I built FlipGate: a release gate that counts per-item answer flips instead of trusting accuracy.

Repo + data below 🧵

**2/** The setup:
- Qwen2.5-3B: bf16 vs AWQ vs GPTQ-Int4
- 200 GSM8K + 541 IFEval + 155 FedProc registry checks
- One RTX 3060, paired stats (McNemar + bootstrap CIs)

**3/** GSM8K says "ship it":
AWQ +11pts, GPTQ +5pts.

FlipGate says "wait":
AWQ broke 13 correct answers (p=0.0024).
GPTQ broke 14.

19% of all items changed answers either way.

**4/** The wildest finding:

Swapping ONLY the engine (HF → llama.cpp, same weights, same precision) flipped 42/200 items.

Engine effect (12%) > quantization effect (7%).

Without an f16 control row I'd have blamed quantization for the engine's flips.

**5/** FedProc hallucination check (no LLM judge, just a clause registry):

AWQ fabricated 34 new FAR clause numbers (p=0.001).
GPTQ fabricated 20 (p=0.038).

Both significant. This is the tripwire that actually fires.

**6/** Lessons:
- Measure a noise floor first (mine: 0%)
- Run an engine control (serving stack ≠ model)
- Report nulls (half my p-values weren't significant — that's data)
- Accuracy hides flips. Count them.

🔗 github.com/raihan-js/flipgate
📊 huggingface.co/datasets/raihan-js/flipgate-results

---
Images to attach:
- Post 1: flipgate_thumbnail.png
- Post 3: flip_directions.png
- Post 4: engine_vs_quantization.png
- Post 5: hallucination_detection.png
---

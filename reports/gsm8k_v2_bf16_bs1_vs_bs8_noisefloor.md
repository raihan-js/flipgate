# FlipGate Report: PASS

- **Baseline:** Qwen2.5-3B-bf16_gsm8k_v2
- **Candidate:** Qwen2.5-3B-bf16_gsm8k_v2_bs8
- **Dataset:** gsm8k
- **Items:** 200

## Accuracy

| Metric | Value |
|--------|-------|
| Baseline accuracy | 79.5% |
| Candidate accuracy | 80.0% |
| Delta | +0.5% |

## Flip Analysis

| Metric | Value |
|--------|-------|
| Right-to-wrong flips | 6 |
| Wrong-to-right flips | 7 |
| Total flips | 13 |
| R-to-W flip rate | 0.0300 |
| 95% CI | [0.0100, 0.0550] |
| McNemar p-value | 1.0000 |

## Flipped Items

### Right-to-wrong (6)

- `item_0021`
- `item_0085`
- `item_0138`
- `item_0186`
- `item_0193`
- `item_0199`

### Wrong-to-right (7)

- `item_0000`
- `item_0046`
- `item_0061`
- `item_0154`
- `item_0156`
- `item_0157`
- `item_0187`

**Verdict: PASS**
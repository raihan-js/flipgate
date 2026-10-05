# FlipGate Report: FAIL

- **Baseline:** Qwen2.5-3B-bf16_gsm8k_v2
- **Candidate:** Qwen2.5-3B-GPTQ-Int4_gsm8k_v2
- **Dataset:** gsm8k
- **Items:** 1000

## Accuracy

| Metric | Value |
|--------|-------|
| Baseline accuracy | 79.7% |
| Candidate accuracy | 76.0% |
| Delta | -3.7% |

## Flip Analysis

| Metric | Value |
|--------|-------|
| Right-to-wrong flips | 91 |
| Wrong-to-right flips | 54 |
| Total flips | 145 |
| R-to-W flip rate | 0.0910 |
| 95% CI | [0.0740, 0.1090] |
| McNemar p-value | 0.0028 |

## Failures

- significant right-to-wrong flip asymmetry

## Flipped Items

### Right-to-wrong (91)

- `item_0005`
- `item_0019`
- `item_0020`
- `item_0021`
- `item_0036`
- `item_0044`
- `item_0048`
- `item_0055`
- `item_0057`
- `item_0070`
- `item_0073`
- `item_0098`
- `item_0114`
- `item_0175`
- `item_0177`
- `item_0182`
- `item_0209`
- `item_0214`
- `item_0235`
- `item_0252`
- `item_0272`
- `item_0277`
- `item_0292`
- `item_0299`
- `item_0307`
- `item_0313`
- `item_0330`
- `item_0347`
- `item_0349`
- `item_0354`
- `item_0361`
- `item_0362`
- `item_0363`
- `item_0373`
- `item_0376`
- `item_0383`
- `item_0409`
- `item_0410`
- `item_0412`
- `item_0422`
- `item_0427`
- `item_0478`
- `item_0482`
- `item_0497`
- `item_0499`
- `item_0506`
- `item_0509`
- `item_0523`
- `item_0527`
- `item_0529`
- `item_0538`
- `item_0540`
- `item_0554`
- `item_0563`
- `item_0587`
- `item_0593`
- `item_0606`
- `item_0623`
- `item_0651`
- `item_0669`
- `item_0683`
- `item_0687`
- `item_0692`
- `item_0706`
- `item_0731`
- `item_0767`
- `item_0771`
- `item_0779`
- `item_0789`
- `item_0790`
- `item_0792`
- `item_0802`
- `item_0815`
- `item_0837`
- `item_0840`
- `item_0849`
- `item_0853`
- `item_0856`
- `item_0857`
- `item_0861`
- `item_0883`
- `item_0900`
- `item_0909`
- `item_0920`
- `item_0923`
- `item_0932`
- `item_0938`
- `item_0960`
- `item_0979`
- `item_0991`
- `item_0997`

### Wrong-to-right (54)

- `item_0000`
- `item_0015`
- `item_0039`
- `item_0043`
- `item_0046`
- `item_0061`
- `item_0108`
- `item_0122`
- `item_0153`
- `item_0154`
- `item_0226`
- `item_0233`
- `item_0249`
- `item_0273`
- `item_0275`
- `item_0297`
- `item_0303`
- `item_0304`
- `item_0357`
- `item_0394`
- `item_0407`
- `item_0419`
- `item_0464`
- `item_0475`
- `item_0489`
- `item_0504`
- `item_0531`
- `item_0546`
- `item_0550`
- `item_0559`
- `item_0577`
- `item_0644`
- `item_0672`
- `item_0707`
- `item_0711`
- `item_0760`
- `item_0796`
- `item_0798`
- `item_0806`
- `item_0810`
- `item_0812`
- `item_0813`
- `item_0846`
- `item_0855`
- `item_0864`
- `item_0877`
- `item_0880`
- `item_0894`
- `item_0914`
- `item_0928`
- `item_0965`
- `item_0972`
- `item_0973`
- `item_0999`

**Verdict: FAIL**
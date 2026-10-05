# FlipGate Report: FAIL

- **Baseline:** Qwen2.5-3B-bf16_gsm8k_v2
- **Candidate:** Qwen2.5-3B-AWQ_gsm8k_v2
- **Dataset:** gsm8k
- **Items:** 1000

## Accuracy

| Metric | Value |
|--------|-------|
| Baseline accuracy | 79.7% |
| Candidate accuracy | 76.2% |
| Delta | -3.5% |

## Flip Analysis

| Metric | Value |
|--------|-------|
| Right-to-wrong flips | 91 |
| Wrong-to-right flips | 56 |
| Total flips | 147 |
| R-to-W flip rate | 0.0910 |
| 95% CI | [0.0730, 0.1090] |
| McNemar p-value | 0.0050 |

## Failures

- significant right-to-wrong flip asymmetry

## Flipped Items

### Right-to-wrong (91)

- `item_0003`
- `item_0005`
- `item_0019`
- `item_0020`
- `item_0021`
- `item_0028`
- `item_0036`
- `item_0054`
- `item_0055`
- `item_0064`
- `item_0075`
- `item_0078`
- `item_0098`
- `item_0111`
- `item_0163`
- `item_0172`
- `item_0182`
- `item_0186`
- `item_0192`
- `item_0193`
- `item_0199`
- `item_0209`
- `item_0260`
- `item_0277`
- `item_0278`
- `item_0287`
- `item_0288`
- `item_0299`
- `item_0316`
- `item_0347`
- `item_0349`
- `item_0361`
- `item_0362`
- `item_0363`
- `item_0383`
- `item_0389`
- `item_0392`
- `item_0395`
- `item_0409`
- `item_0412`
- `item_0422`
- `item_0427`
- `item_0436`
- `item_0467`
- `item_0493`
- `item_0497`
- `item_0509`
- `item_0526`
- `item_0528`
- `item_0538`
- `item_0545`
- `item_0554`
- `item_0562`
- `item_0563`
- `item_0565`
- `item_0576`
- `item_0583`
- `item_0593`
- `item_0606`
- `item_0609`
- `item_0669`
- `item_0681`
- `item_0687`
- `item_0694`
- `item_0696`
- `item_0704`
- `item_0727`
- `item_0728`
- `item_0740`
- `item_0767`
- `item_0771`
- `item_0779`
- `item_0790`
- `item_0791`
- `item_0792`
- `item_0815`
- `item_0825`
- `item_0837`
- `item_0840`
- `item_0844`
- `item_0909`
- `item_0922`
- `item_0923`
- `item_0932`
- `item_0960`
- `item_0961`
- `item_0963`
- `item_0967`
- `item_0969`
- `item_0979`
- `item_0990`

### Wrong-to-right (56)

- `item_0000`
- `item_0015`
- `item_0037`
- `item_0039`
- `item_0061`
- `item_0093`
- `item_0099`
- `item_0108`
- `item_0124`
- `item_0154`
- `item_0216`
- `item_0226`
- `item_0233`
- `item_0234`
- `item_0245`
- `item_0250`
- `item_0273`
- `item_0285`
- `item_0297`
- `item_0304`
- `item_0323`
- `item_0340`
- `item_0357`
- `item_0369`
- `item_0407`
- `item_0411`
- `item_0413`
- `item_0419`
- `item_0489`
- `item_0504`
- `item_0505`
- `item_0531`
- `item_0539`
- `item_0546`
- `item_0610`
- `item_0618`
- `item_0644`
- `item_0700`
- `item_0701`
- `item_0707`
- `item_0711`
- `item_0721`
- `item_0724`
- `item_0760`
- `item_0806`
- `item_0810`
- `item_0814`
- `item_0855`
- `item_0877`
- `item_0894`
- `item_0912`
- `item_0928`
- `item_0948`
- `item_0972`
- `item_0973`
- `item_0984`

**Verdict: FAIL**
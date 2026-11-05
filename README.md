# Map the Flow: reproduction and pathway ablation

This repository packages a four-task TVBench reproduction of *Map the Flow* (Kim, Kim, Han; ICLR 2026) with the released LLaVA-NeXT-7B-Video-FT model. It also tests shifted, narrowed, and removed attention pathways on a fixed 448-question subsample. The JSONL results contain answer and score fields only; the scripts fetch the original code, model, and dataset separately.

## Results

Reproduction accuracy (%):

| Source | Full | Effective | Random |
| --- | ---: | ---: | ---: |
| Mine (4 tasks, n=1,002) | 50.5 | 48.3 | 40.4 |
| Paper Table 3 (full TVBench) | 51.5 | 51.2 | 40.1 |

Pooled ablation (3 tasks, n=448):

| Condition | Accuracy (%) | Mean %dp | Edges kept (%) |
| --- | ---: | ---: | ---: |
| Full | 58.0 | 0.0 | 100.0 |
| Effective | 55.8 | -1.3 | 43.1 |
| Random42 | 41.7 | -42.2 | 42.0 |
| Shift -3 | 54.2 | +0.1 | 43.1 |
| Shift +3 | 47.5 | -18.2 | 43.1 |
| Shrink | 48.0 | -13.2 | 33.3 |
| Random33 | 38.2 | -53.7 | 33.3 |
| Drop inter-frame | 45.5 | -21.6 | 20.5 |
| Drop video→question | 40.4 | -34.7 | 40.4 |
| Drop question→last | 41.1 | -31.4 | 43.1 |

### How to read %dp

Mean %dp is the mean relative change in the baseline answer's probability, computed only over questions the unmasked baseline answered correctly: `(masked_probability - baseline_probability) / baseline_probability × 100`. More negative values indicate a larger loss of confidence in the baseline answer, even when accuracy changes little.

## Caveats

The ablation has 448 questions, one model, and one subsample seed; accuracy gaps of 1–2 points are noise, so %dp is the main metric. Shift +3 overlaps the fixed vision→vision (from L21) and question→question (from L26) blocks, so it is not a pure shift. Moving Direction is excluded from the ablation because the unmasked model is at chance. Novelty relative to the paper and follow-ups has not been checked. The reproduction's effective-pathway accuracy drop is 2.2 points versus 0.3 in the paper; its cause has not been tested.

## Reproduce

See [scripts/README.md](scripts/README.md) for the exact server sequence and result export. `pathway_ablation/check_vs_gate2.py` is optional and requires original, unstripped upstream JSON output.

## Layout

```text
pathway_ablation/             Conditions, runner, summary, optional check
scripts/                      Server setup, runs, filtered export
results/ablation/             42% random and pathway ablation JSONL + table
results/ablation_random33/    33% random control JSONL + table
results/reproduction/         Filtered four-task JSONL + table
NOTICE.md                     Credits and provenance
```

## Credits

See [NOTICE.md](NOTICE.md).

# Benchmark: held-out genera on SILVA and PR2

ATLAS exists for reads from organisms missing from the reference database. A
test split of the reference cannot measure that, because every genus in it
was also in training. So this benchmark removes whole genera before training
and asks two questions:

1. On genera the model knows, how accurate is it, and does its calibrated
   threshold keep the precision it promises?
2. On genera it has never seen, how many reads does the Filter refuse, and
   does the Explorer group them by their true genus?

The raw results are in [benchmark/](benchmark/) (`16S-silva.json`,
`18S-pr2.json`) and were produced by `atlas evaluate` through
[scripts/benchmark.sh](../scripts/benchmark.sh).

## Setup

| | 16S | 18S |
|---|---|---|
| Reference | SILVA 138.2 SSU Ref NR99 (bacteria and archaea) | PR2 5.1.1 SSU, `taxo_long` (eukaryotes) |
| Rank | genus | genus |
| Classes kept | 10 or more sequences, capped at 20 each | same |
| Held out | 50 genera chosen at random (seed 42) | same |
| Known genera, sequences | 2,427 genera, 44,146 sequences | 2,302 genera, 39,290 sequences |
| Test sequences of known genera | 6,739 (15% per genus) | 6,062 |
| Sequences of held-out genera | 941 | 817 |

Placeholder names (`uncultured`, `Incertae_sedis`, PR2's `_X` levels,
`sp.`) are removed from lineages before labelling, so no class is a
placeholder.

Model: 6-mer profiles (4,096 features), an MLP with 1024 and 512 ReLU units
and dropout 0.5, Adam at 1e-3, batch 64, early stopping on validation loss
with patience 5. The confidence threshold is calibrated on the validation
split for 95% precision. Explorer: PCA, then HDBSCAN with a minimum cluster
size of 5.

Hardware and software: a Docker container (python:3.12-slim) with 16 logical
CPUs and 8 GB of memory, no GPU; TensorFlow 2.21, Keras 3.15, scikit-learn
1.9, NumPy 2.5. TensorFlow on CPU is not bit-for-bit deterministic, so a
rerun gives slightly different numbers.

## Genera the model knows

| | 16S | 18S |
|---|---|---|
| Accuracy | 92.1% | 86.4% |
| Macro-F1 | 0.915 | 0.845 |
| Calibrated threshold | 0.53 | 0.70 |
| Reads kept at the threshold | 95.4% | 81.3% |
| Precision of the kept reads | 94.9% | 95.1% |
| Epochs until early stopping, training time | 35, 10.3 min | 27, 7.0 min |

The threshold is chosen on the validation split and checked on the test
split: kept reads are 94.9% and 95.1% correct against a 95% target. 18S
needs a stricter threshold than 16S to reach the same precision, so it keeps
fewer reads.

## Genera the model has never seen

| | 16S | 18S |
|---|---|---|
| Reads refused by the Filter at the calibrated threshold | 31.4% (295 of 941) | 70.5% (576 of 817) |
| Known vs novel separation by confidence (AUROC) | 0.861 | 0.849 |
| Refused reads placed in an Explorer cluster | 89.8% | 89.8% |
| Explorer clusters | 29 | 54 |
| Agreement of clusters with the true genus (adjusted Rand index) | 0.905 | 0.836 |
| Homogeneity (clusters rarely mix genera) | 0.957 | 0.968 |
| Held-out genera recovered: a cluster that is at least 80% that genus | 25 of 50 | 40 of 50 |
| Clusters flagged as putative novel lineages | 3 of 29 | 3 of 54 |

The Explorer does its part well: the reads it receives are grouped by their
true genus with little mixing, and on 18S 40 of the 50 unseen genera come out
as a cluster of their own.

The weak link is the hand-off. The Filter's confidence is a softmax
probability, and a network is often confident on inputs unlike anything it
was trained on. On 16S, 69% of the reads from unseen genera were assigned to
a known genus with enough confidence to pass the threshold, so they never
reached the Explorer, and only 25 of the 50 genera could be recovered. The
AUROC of about 0.85 says confidence does separate known from novel reads on
average, but the two distributions overlap. 16S is harder than 18S here; a
likely reason is that a held-out bacterial genus often has a close relative
among 2,400 known genera, but this benchmark does not measure that directly.

The novelty flag is conservative. It fires when a cluster is farther from
every known genus than 95% of reference sequences are from their own genus,
and most held-out genera are not that far from their relatives. An unflagged
cluster means "unassigned", not "known".

## The trade-off: choosing a threshold

A higher threshold refuses more novel reads and keeps fewer known ones, with
higher precision on what it keeps. From the same runs:

**16S (SILVA)**

| Threshold | Known reads kept | Precision of kept | Novel reads refused |
|---|---|---|---|
| 0.50 | 96.2% | 94.5% | 28.4% |
| 0.60 | 93.4% | 95.8% | 39.9% |
| 0.70 | 90.3% | 96.8% | 51.1% |
| 0.80 | 86.5% | 97.6% | 62.3% |
| 0.90 | 81.1% | 98.7% | 73.4% |
| 0.95 | 76.3% | 99.1% | 80.5% |
| 0.99 | 63.1% | 99.6% | 92.0% |

**18S (PR2)**

| Threshold | Known reads kept | Precision of kept | Novel reads refused |
|---|---|---|---|
| 0.50 | 90.8% | 91.5% | 49.5% |
| 0.60 | 86.1% | 93.5% | 60.6% |
| 0.70 | 81.4% | 95.0% | 70.3% |
| 0.80 | 76.5% | 96.3% | 79.4% |
| 0.90 | 68.9% | 97.6% | 86.4% |
| 0.95 | 61.9% | 98.3% | 90.3% |
| 0.99 | 48.1% | 99.2% | 95.6% |

For a deep-sea sample, where unknown taxa are the point, a threshold around
0.9 is a reasonable choice: most known reads are still classified, at higher
precision, and roughly three quarters or more of novel reads reach the
Explorer. Use `atlas analyze --threshold 0.9` for one run, or
`atlas train --target-precision 0.99` to calibrate a stricter threshold into
the model.

Better open-set methods (distance to class centroids in the network's
embedding, energy scores, or training with an explicit "other" class of
out-of-group sequences) would move this curve and are the obvious next step.

## Learning rate

ATLAS v1 trained with Adam at 1e-4, a change made for v1's small datasets.
On the full 16S benchmark:

| Learning rate | Epochs | Time | Accuracy | Macro-F1 | Kept at threshold, precision | Novel refused |
|---|---|---|---|---|---|---|
| 1e-4 | 100 (the cap, still improving) | 55.8 min | 91.5% | 0.911 | 95.1% at 94.4% | 33.0% |
| 1e-3 | 35 (early stopping) | 10.3 min | 92.1% | 0.915 | 95.4% at 94.9% | 31.4% |

1e-3 reaches a slightly better model in a fifth of the time, so it is the
default in v2; `--learning-rate 1e-4` is still available. The 1e-4 result is
in [benchmark/16S-silva-lr1e-4.json](benchmark/16S-silva-lr1e-4.json) (that
run predates the AUROC and threshold sweep).

## Reproduce

```bash
pip install -e .
scripts/benchmark.sh
```

The script downloads SILVA and PR2 (about 260 MB) into `benchmark-data/` and
writes the JSON results to `docs/benchmark/`. It takes about 20 minutes on 16
CPU cores and fits in 8 GB of memory.

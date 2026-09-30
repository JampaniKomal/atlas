# Training marker models

ATLAS classifies with one model per marker gene. Each is trained from a public
reference database that you download once; neither the databases nor trained
models are stored in this repository.

| Marker | Organisms | Reference | Parser | Download |
|---|---|---|---|---|
| 16S | bacteria, archaea | SILVA SSU Ref NR99 | `silva` | [arb-silva.de](https://www.arb-silva.de/download/arb-files/) (`SILVA_138.2_SSURef_NR99_tax_silva.fasta.gz`, 200 MB) |
| 18S | eukaryotes | PR2 | `pr2` | [PR2 releases](https://github.com/pr2database/pr2database/releases) (`pr2_version_5.1.1_SSU_taxo_long.fasta.gz`, 60 MB) |
| COI | animals | MIDORI2 | `midori` | [reference-midori.info](https://www.reference-midori.info/) (a `*_CO1_*` FASTA with taxonomy) |
| ITS | fungi | UNITE general release | `unite` | [unite.ut.ee](https://unite.ut.ee/repository.php) |

SILVA can also serve 18S: pass `--marker 18S --parser silva`, and only
sequences whose lineage starts with Eukaryota are kept.

## Train

```bash
atlas train --marker 16S --reference SILVA_138.2_SSURef_NR99_tax_silva.fasta.gz \
            --min-members 10 --max-per-class 30
atlas info
```

What happens:

1. Headers are parsed into lineages; placeholder names (`uncultured`,
   `Incertae_sedis`, PR2's `_X` levels) are dropped, and only the marker's
   domains are kept.
2. Sequences are labelled at `--rank` (genus by default). Classes with fewer
   than `--min-members` sequences are removed, and large classes are capped
   at `--max-per-class` so a few well-studied genera don't dominate.
3. Each sequence becomes a 4,096-dimensional 6-mer profile.
4. The data is split per class into 70% training, 15% validation and 15%
   test. The network trains with Adam (learning rate 1e-3) until validation
   loss hasn't improved for 5 epochs, then keeps the best weights.
5. A confidence threshold is calibrated on the validation split: the lowest
   cut-off at which accepted predictions are at least 95% correct
   (`--target-precision`). Reads below it go to the Explorer.
6. Accuracy, macro-F1, and coverage and precision at the threshold are
   measured on the test split and stored with the model.

The result is `models/<marker>/` with `model.keras`, `meta.json` and
`centroids.npy`. Train as many markers as you have data for;
`atlas analyze` routes each read to the marker it resembles.

## Useful options

| Option | Default | When to change it |
|---|---|---|
| `--rank` | genus | `family` for a coarser, more robust classifier; `species` if the reference supports it |
| `--min-members` | 3 | raise it (10 or more) for cleaner classes on large references |
| `--max-per-class` | none | cap it to balance classes and bound memory |
| `--k` | 6 | 7 or 8 for long, conserved markers (memory grows 4× per step) |
| `--hidden` | 1024 512 | smaller for small references |
| `--learning-rate` | 1e-3 | 1e-4 (the v1 setting) for small references where training is unstable |
| `--epochs` | 100 | early stopping usually ends much sooner (27 to 35 epochs on SILVA and PR2) |
| `--target-precision` | 0.95 | raise it (0.99) to refuse more reads from unknown genera, at the cost of coverage |

Memory: the training matrix holds sequences × 4^k float32 values, about
16 KB per sequence at k = 6. 100,000 sequences need about 1.6 GB.

A GPU is used automatically when TensorFlow finds one (Linux, or Windows
through WSL2); see [INSTALL.md](INSTALL.md).

## Measure it on your data

```bash
atlas evaluate --marker 18S --reference pr2_version_5.1.1_SSU_taxo_long.fasta.gz --holdout 20
```

This removes 20 genera before training and reports how the model does on the
genera it knows and how the Explorer handles the ones it doesn't. See
[BENCHMARK.md](BENCHMARK.md) for results on SILVA and PR2.

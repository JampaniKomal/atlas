# ATLAS: Artificial Taxonomic Learning & Analysis System

[![CI](https://github.com/JampaniKomal/atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/JampaniKomal/atlas/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Taxonomy and biodiversity from environmental DNA, including the organisms
your reference database has never seen.**

ATLAS reads a FASTA file of eDNA sequences (16S, 18S, COI or ITS) and:

1. **Filter**: classifies each sequence to genus with a deep-learning model
   per marker gene, and keeps only predictions above a confidence threshold
   calibrated for 95% precision;
2. **Explorer**: clusters the sequences no model is sure about (HDBSCAN over
   k-mer composition), and flags clusters that are farther from every known
   genus than known sequences are from their own: candidate novel lineages;
3. **Biodiversity**: estimates abundance and alpha diversity (observed units,
   Chao1, Shannon, Gini-Simpson, Pielou) over classified genera and Explorer
   clusters together;
4. reports it as text, JSON and a standalone HTML page, from the command line
   or a local web interface.

Built for Smart India Hackathon 2025, problem statement **SIH25042**:
*Identifying taxonomy and assessing biodiversity from eDNA datasets*, posed by
the Centre for Marine Living Resources and Ecology (CMLRE).

## The problem

Deep-sea sediment and water samples carry DNA from organisms nobody has
sequenced before. Standard pipelines (QIIME 2, DADA2, mothur) assign reads by
aligning them to reference databases such as SILVA, PR2 or NCBI, which are
dominated by well-studied terrestrial and shallow-water species. Deep-sea
reads end up misclassified, dropped as "unassigned", or forced into the
nearest known genus, and biodiversity is underestimated.

ATLAS treats that gap as a first-class output. A classifier trained on the
reference handles what it knows. Anything it isn't confident about goes to
unsupervised clustering instead of being discarded or forced into a label, so
an unknown lineage shows up as a cluster with a representative sequence to
investigate.

## How it works

```
FASTA ──► 6-mer profiles ──► route to marker (nearest class centroid)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
      Filter: marker MLP (Keras)        below threshold
      confidence ≥ calibrated cut-off        │
                  │                          ▼
                  │            Explorer: PCA ─► HDBSCAN ─► clusters
                  │            nearest known genus, similarity, novelty flag
                  ▼                          ▼
          genus abundance  ───────►  diversity (units = genera + clusters)
                                             │
                                             ▼
                              text / JSON / HTML report, web UI
```

- **Features.** Each sequence becomes a 4,096-dimensional 6-mer profile,
  counted with vectorised numpy (2-bit base codes, base-4 k-mer indices) and
  L2-normalised, so profiles of different lengths are comparable and a dot
  product is a cosine similarity.
- **Filter.** One multilayer perceptron per marker (1024 and 512 ReLU units,
  dropout 0.5, Adam at 1e-3, early stopping with patience 5), trained on a
  70/15/15 stratified split. The confidence threshold is the lowest cut-off
  at which validation predictions are at least 95% correct.
- **Explorer.** Unclassified profiles are reduced with PCA and clustered with
  HDBSCAN, which finds clusters of different densities and leaves outliers
  unclustered rather than forcing them into a group. Each cluster is
  described by its nearest known genus, the cosine similarity to it, and a
  representative sequence (the member closest to the cluster centre) to
  confirm with BLAST or phylogenetic placement.
- **Novelty flag.** During training ATLAS records how similar known sequences
  are to their own genus centroid. A cluster below the 5th percentile of that
  distribution is more distant from every known genus than 95% of reference
  sequences are from their own, and is reported as a putative novel lineage.

## Quick start

```bash
git clone https://github.com/JampaniKomal/atlas && cd atlas
pip install -e .
atlas demo
```

`atlas demo` builds a synthetic reference and a 600-read community that
includes two genera absent from the reference, trains a model, analyses the
sample and writes `atlas-demo/report.html`. It takes under a minute on a
laptop CPU:

```
[  PART 2: EXPLORER RESULTS  ]
| Unclassified: 87 reads -> 4 clusters, 0 unclustered
|   16S-X1      27 reads  nearest GenusA1 (similarity 0.280)  PUTATIVE NOVEL LINEAGE
...
[  PART 3: BIODIVERSITY  ]
| Observed units (taxa + clusters): 16    Chao1 estimate: 16.0
| Shannon H': 2.491    Gini-Simpson: 0.8956    Pielou evenness: 0.8984

Against the known truth: 513/513 known reads classified correctly; 87/87 reads
from absent genera left unclassified; 87 reads in clusters flagged as putative
novel lineages.
```

With real data:

```bash
atlas train --marker 18S --reference pr2_version_5.1.1_SSU_taxo_long.fasta.gz --min-members 10 --max-per-class 20
atlas analyze sample.fasta                  # text report, plus reports/<name>.html
atlas analyze sample.fasta --threshold 0.9  # stricter: more reads go to the Explorer
atlas analyze                               # asks for the file interactively
atlas serve                                 # web interface on http://127.0.0.1:5000
```

Reference databases, parsers and training options for all four markers are in
[docs/TRAINING.md](docs/TRAINING.md); installation (pip, conda, Docker, GPU)
in [docs/INSTALL.md](docs/INSTALL.md).

## Results on real reference databases

The held-out-genera benchmark removes 50 whole genera from the reference
before training, then tests on sequences of the remaining genera and on every
sequence of the removed ones, which the model has never seen. Genus level,
6-mers, classes of 10 or more sequences capped at 20, on a 16-CPU container
without a GPU. Details, the full threshold sweep and how to reproduce it are
in [docs/BENCHMARK.md](docs/BENCHMARK.md).

| | 16S, SILVA 138.2 | 18S, PR2 5.1.1 |
|---|---|---|
| Known genera | 2,427 | 2,302 |
| Closed-set accuracy (macro-F1) | 92.1% (0.915) | 86.4% (0.845) |
| Kept at the calibrated threshold, and precision of what is kept | 95.4% at 94.9% | 81.3% at 95.1% |
| Reads from held-out genera refused by the Filter | 31% | 71% |
| Known vs novel separation by confidence (AUROC) | 0.86 | 0.85 |
| Explorer: clustered novel reads vs true genus (ARI, homogeneity) | 0.90, 0.96 | 0.84, 0.97 |
| Held-out genera recovered as their own cluster | 25 of 50 | 40 of 50 |
| Training time | 10 min, 35 epochs | 7 min, 27 epochs |

What this says: on genera it knows, the Filter is accurate, and what it keeps
is about 95% correct, the precision its threshold is calibrated for. On genera it doesn't know, confidence alone
is a weak detector, especially for 16S: two thirds of the reads from
held-out bacterial genera were confidently assigned to a known genus. The
reads that do reach the Explorer are grouped by their true genus with little
mixing. A stricter threshold trades coverage for novelty detection: at 0.9,
16S refuses 73% of novel reads and keeps 81% of known ones at 98.7%
precision (`atlas analyze --threshold 0.9`, or train with
`--target-precision 0.99`).

## Project layout

```
atlas/
  kmers.py       vectorised k-mer profiles
  fasta.py       streaming FASTA reader (plain or gzip)
  references.py  SILVA, PR2, UNITE and MIDORI2 header parsers
  markers.py     per-marker defaults (16S, 18S, COI, ITS)
  dataset.py     labelled training sets and stratified splits
  model.py       Filter: Keras MLP, threshold calibration, metrics
  explorer.py    Explorer: PCA + HDBSCAN, nearest-genus annotation, novelty flag
  diversity.py   richness, Chao1, Shannon, Gini-Simpson, Pielou
  analysis.py    routing, Filter, Explorer and diversity for one sample
  report.py      text, JSON and HTML reports
  server.py      `atlas serve` (Flask) for the web interface in web/index.html
  evaluate.py    held-out-genera benchmark
  synthetic.py   synthetic references and communities (demo and tests)
  cli.py         the `atlas` command
tests/           unit tests and an end-to-end test of train, analyse and serve
docs/            installation, training, benchmark, and v1 history
```

## History

ATLAS was built in September 2025 for SIH 2025. Development went
notebook-first: each marker's data preparation and training were worked out in
Jupyter, then turned into scripts (16S from SILVA, 18S, COI with a hashed
k-mer vectoriser, ITS for fungi), with a Doc2Vec + HDBSCAN "Explorer" for
unclassified reads. The front end went through a web page with a Flask
backend and an Electron desktop build before settling on a command-line tool.
The development logs are in [docs/history](docs/history/).

Version 2 (2026) turned the scripts into one installable package and command,
replaced per-k-mer Python dictionaries with vectorised counting, calibrates
the confidence threshold instead of fixing it at 0.8, clusters k-mer profiles
directly (no second model, deterministic), adds diversity estimates and HTML
reports, brings the original web interface back to life with `atlas serve`,
and measures everything on real databases with a held-out-genera benchmark.

## Limitations

- ATLAS classifies near-full-length marker sequences best, because that is
  what reference databases contain. Short amplicon reads (for example 16S
  V4, about 250 bp) should be classified with a model trained on the same
  region of the reference.
- The Filter's confidence is a softmax probability, and a network is often
  confident on inputs unlike anything it was trained on. In the benchmark it
  refused only 31% (16S) and 71% (18S) of reads from genera it had never
  seen at the default threshold; the rest were assigned to a known genus.
  Raise the threshold when unknown taxa matter more than coverage.
- The novelty flag is conservative: 3 of 29 (16S) and 3 of 54 (18S) Explorer
  clusters of held-out genera were flagged, because most held-out genera
  have close relatives in the reference. Treat unflagged clusters as
  unassigned, not as known.
- The novelty flag measures distance from the reference in k-mer space. It
  points at sequences worth investigating; naming a new taxon needs
  phylogenetics and expert review.
- Genus-level labels are only as good as the reference taxonomy, and genera
  with fewer than `--min-members` reference sequences can't be learnt.
- Abundance is read counts, not organism counts; copy-number and PCR biases
  are not corrected.

## Contributors

- **Jampani Komal**: data pipelines, Filter and Explorer, CLI, web interface,
  and version 2.
- **Rishu Tiwari**: model training improvements (the longer early-stopping
  patience ATLAS still uses, and the lower Adam learning rate v1 trained
  with) and the HashingVectorizer, generator-based k-mer pipeline for COI.

## License

MIT. See [LICENSE](LICENSE). Reference databases have their own licences and
citation requirements; follow them when you publish results.

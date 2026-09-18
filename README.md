# ATLAS: Artificial Taxonomic Learning & Analysis System

This repository contains the official codebase for the ATLAS project, an AI-driven software suite for taxonomic identification and biodiversity assessment from environmental DNA (eDNA).

## About This Repository
This repository implements an AI-driven pipeline that minimizes reliance on reference databases, reduces computational time, and enables the discovery of novel taxa and ecological insights in deep-sea environments.

## Project Mission

The goal of ATLAS is to address a critical challenge in modern biodiversity research: the "database gap." Standard reference databases are often incomplete, especially for organisms from unique biomes like the deep sea. ATLAS is an AI-driven pipeline that minimizes reliance on these databases, reduces computational time, and enables the discovery of novel taxa from raw eDNA reads.

For a detailed overview of the project's scientific background and long-term goals, please see the [Project Overview](docs/01_Project_Overview.md) document.

## Problem Statement & Context

**Problem Statement ID:** ID25042  
**Problem Statement Title:** Identifying Taxonomy and Assessing Biodiversity from eDNA Datasets

### Description

The deep ocean, encompassing vast and remote ecosystems like abyssal plains, hydrothermal vents, and seamounts, harbors a significant portion of global biodiversity, much of which remains undiscovered due to its inaccessibility. Understanding deep-sea biodiversity is critical for elucidating ecological interactions (e.g., food webs, nutrient cycling), informing conservation strategies for vulnerable marine habitats, and identifying novel eukaryotic species with potential biotechnological or ecological significance.

Environmental DNA (eDNA) has emerged as a powerful, non-invasive tool for studying these ecosystems by capturing genetic traces of organisms from environmental samples, such as seawater or sediment, without the need for physical collection or disturbance of fragile habitats. By targeting marker genes like 18S rRNA or COI, eDNA enables the detection of diverse eukaryotic taxa, including protists, cnidarians, and rare metazoans, offering insights into species richness and community structure.

The Centre for Marine Living Resources and Ecology (CMLRE) will undertake routine voyages to the deep sea and collect sediment and water samples from hotspot regions for biodiversity assessment and ecosystem monitoring. The water and sediment samples will be used to extract eDNA and will be subject to high-throughput sequencing.

However, assigning raw eDNA sequencing reads to eukaryotic taxa or inferring their ecological roles presents significant challenges, primarily due to the poor representation of deep-sea organisms in reference databases like SILVA, PR2, or NCBI. These databases, built primarily from well-studied terrestrial or shallow-water species, lack comprehensive sequences for deep-sea eukaryotes, leading to misclassifications, unassigned reads, or underestimation of biodiversity.

Traditional bioinformatic pipelines for eDNA analysis, such as those implemented in QIIME2, DADA2, or mothur, rely heavily on sequence alignment or mapping to these databases, which is inadequate for novel or divergent deep-sea taxa. This dependency limits the discovery of new species and hinders accurate biodiversity assessments, critical for conservation in rapidly changing deep-sea environments. The computational time required for processing eDNA data exacerbates these challenges, particularly given the limitations of database-dependent methods and the complexity of eDNA datasets.

### Expected Solution

To address the challenges of poor database representation and computational time in deep-sea eDNA analysis, we propose an AI-driven pipeline that uses deep learning and unsupervised learning to identify eukaryotic taxa and assess biodiversity directly from raw eDNA reads. The solution should be able to classify the sequences, annotate and estimate abundance. This solution minimizes reliance on reference databases, reduces computational time through optimized workflows, and enables the discovery of novel taxa and ecological insights in deep-sea ecosystems.

## Getting Started

To get started with ATLAS, you will need to set up a Conda environment and download the required reference databases.

### 1. Environment Setup

This project supports both GPU-accelerated and CPU-only workflows. Please follow the [Environment and Installation Guide](docs/02_Environment_and_Installation.md) for detailed, step-by-step instructions on setting up the correct environment for your system.

### 2. Data Acquisition

The ATLAS pipelines rely on large, public reference databases (e.g. SILVA
for 16S) that are not included in this repository. You must download them
manually and place them in `data/raw/`. See the [16S Pipeline Workflow](docs/03_Pipeline_16S_Workflow.md)
for the expected input file for that marker.

### 3. Training a Pipeline

Once your environment is configured and the data is in place, you can train
any of the four marker classifiers (16S, 18S, COI, ITS). Each pipeline
consists of a two-step script-based workflow. For example, to train the 16S
classifier:

```bash
# First, run the data preparation script
python src/pipeline_16s/01_prepare_data.py

# Then, run the model training script
python src/pipeline_16s/02_train_model.py
```

Repeat for `pipeline_18s`, `pipeline_coi`, and `pipeline_its` to train the
remaining Filter models, and see `src/pipeline_explorer/` for the Doc2Vec +
HDBSCAN pipeline used to cluster sequences none of the Filter models can
classify.

### 4. Running an Analysis

Once at least one marker's model artifacts exist under `models/`, analyze a
FASTA file with the CLI:

```bash
python -m cli --input_fasta path/to/your/file.fasta
```

Or run it with no arguments for an interactive prompt. Add `--verbose` to
see per-stage progress, and `--report-name <name>` to control the saved
report's filename (reports are written to `reports/`).

For detailed workflow instructions for each pipeline, please refer to the
documentation in the [docs](docs/) directory.

## Documentation

The project includes documentation in the `docs/` directory:

1. [Project Overview](docs/01_Project_Overview.md) - Scientific background and project goals
2. [Environment and Installation Guide](docs/02_Environment_and_Installation.md) - Setup instructions
3. [16S Pipeline Workflow](docs/03_Pipeline_16S_Workflow.md) - Step-by-step 16S (prokaryote) data preparation workflow
4. [16S Development Log](docs/04_Development_Log_16S.md) - Strategic decisions and technical challenges, 16S
5. [18S Development Log](docs/05_Development_Log_18S.md) - Strategic decisions and technical challenges, 18S (eukaryote)
6. [COI Development Log](docs/06_Development_Log_COI.md) - Strategic decisions and technical challenges, COI (animalia)
7. [ITS Development Log](docs/07_Development_Log_ITS.md) - Strategic decisions and technical challenges, ITS (fungi)

Only the 16S pipeline has a dedicated step-by-step workflow doc; 18S/COI/ITS
are documented as development logs (decisions and challenges) rather than
workflow guides, since all four pipelines follow the same overall shape
(prepare data -> train model) described in `src/pipeline_<marker>/`.

## Project Structure

- **`cli/`**: The interactive command-line entry point (`python -m cli`).
- **`src/`**: Production Python scripts — `predict.py` (the master analysis
  engine used by the CLI), and one `pipeline_<marker>/` folder per genetic
  marker (16S, 18S, COI, ITS) plus `pipeline_explorer/` for the novel-taxa
  clustering pipeline.
- **`docs/`**: Project documentation and per-marker development logs.
- **`others/`**: Standalone utility scripts (e.g. generating a small test
  FASTA file from a full reference database).
- **`data/`** and **`models/`**: Not included in the repo (gitignored —
  reference databases and trained model artifacts are too large to commit).
  Created locally when you download the reference data and run a pipeline's
  training scripts.

## Testing & Verification

Model artifacts (the trained `.keras` classifiers and Doc2Vec model) aren't
included in the repo and require downloading large reference databases to
train, so full end-to-end classification wasn't runnable in this pass.
Instead, verified the pipeline logic directly against real, controlled
inputs, which surfaced 3 real bugs:

- **The "Explorer" (novel taxa discovery) pipeline crashed on every
  genuinely novel sequence.** `explorer_step_1_vectorize()` looked up
  vectors via `doc2vec_model.dv[seq.id]`, which only works for sequence IDs
  that were part of the Doc2Vec model's *training* corpus — copied from the
  training script (`pipeline_explorer/01_vectorize_sequences.py`), where
  that assumption is correct, into the inference-time code, where it isn't:
  every sequence reaching this stage is by definition one the Filter models
  couldn't classify, so its ID was never seen during training. Reproduced
  with a real trained Doc2Vec model and a real `KeyError` on a novel ID,
  then fixed by switching to `doc2vec_model.infer_vector()` (the correct
  Doc2Vec API for embedding unseen documents) and re-verified the same
  scenario now vectorizes, clusters, and reports correctly.
- **`src/predict.py`'s own CLI entry point crashed immediately in
  interactive mode** with `NameError: name 'ATLAS_ASCII' is not defined` —
  the constant is defined in `cli/__main__.py` but `predict.py` has its own
  duplicate `__main__` block that references it without defining it.
  Fixed by defining it in `predict.py` too.
- **A model-loading failure would crash instead of logging cleanly**:
  `logging.error(..., file=sys.stderr)` — `file=` isn't a valid keyword for
  the `logging` module (it's a `print()` kwarg) — so any real error loading
  a model's artifacts raised a `TypeError` instead of the intended clean
  error message. Fixed at both call sites.

Also fixed `others/create_test_fasta.py`, which had the original
developer's own machine-specific absolute paths (`C:\Users\...\Music\atlas\...`)
hardcoded — non-portable for anyone else running it. Now resolves paths
relative to the project root like every other script in the repo.

## Known Limitations

- `index.html` is a full web UI (file upload, live charts, results screen)
  left over from an earlier architecture — it expects a `POST /run_analysis`
  backend that was deliberately removed (`server.py`) in favor of the CLI.
  It is not linked from anywhere (no GitHub Pages) and does not work as
  committed; **the CLI (`python -m cli`) is the actual supported interface.**
- The "Explorer" pipeline's clustering quality depends heavily on how well
  the pre-trained Doc2Vec model generalizes to genuinely novel sequences via
  `infer_vector()` — this is inherently approximate compared to vectors for
  documents seen during training.
- No automated test suite; verification here was done with standalone
  scripts exercising the real pipeline functions with synthetic/controlled
  data, not a committed `tests/` directory.
- GPU support requires a specific CUDA Toolkit/cuDNN version pinned to
  TensorFlow 2.10 (see the Environment and Installation Guide); newer GPU
  drivers may need a different pinned combination.

## Contributors

- **Jampani Komal** — primary development (data pipelines, CLI, Filter and
  Explorer pipelines).
- **Rishu Tiwari** — model training improvements: reduced Adam learning
  rate and increased EarlyStopping patience on the 16S/18S/ITS classifiers,
  and rewrote COI's data preparation to use `HashingVectorizer` with a
  generator-based k-mer workflow for better performance on large datasets.

## License

MIT — see [LICENSE](LICENSE).


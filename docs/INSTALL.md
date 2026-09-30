# Installing ATLAS

ATLAS needs Python 3.11 or newer. TensorFlow is the only large dependency.

## pip

```bash
git clone https://github.com/JampaniKomal/atlas && cd atlas
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
atlas demo
```

## Conda

```bash
conda env create -f environment.yml
conda activate atlas
atlas demo
```

## Docker

```bash
docker build -t atlas .
docker run --rm atlas demo
docker run --rm -v "$PWD:/work" atlas train --marker 16S --reference SILVA_138.2_SSURef_NR99_tax_silva.fasta.gz
docker run --rm -p 5000:5000 -v "$PWD/models:/work/models" atlas serve --host 0.0.0.0
```

## GPU

TensorFlow 2.16 and later use an NVIDIA GPU automatically on Linux when the
driver is installed; `pip install "tensorflow[and-cuda]"` brings the matching
CUDA libraries. On Windows, run ATLAS inside WSL2 for GPU support (native
Windows builds of TensorFlow are CPU-only). The analysis report states whether
a GPU was used.

ATLAS v1 pinned TensorFlow 2.10 with `cudatoolkit=11.2` and `cudnn=8.1.0`,
the last combination with native Windows GPU support; that setup is described
in [history/](history/).

## Check the installation

```bash
pytest -q          # 18 tests, under a minute on a laptop CPU
atlas demo         # trains on a synthetic community and writes atlas-demo/report.html
```

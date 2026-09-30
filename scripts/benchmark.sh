#!/usr/bin/env bash
# Reproduces docs/BENCHMARK.md: the held-out-genera benchmark on SILVA (16S)
# and PR2 (18S). Downloads about 260 MB into benchmark-data/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p benchmark-data
cd benchmark-data
[ -f SILVA_138.2_SSURef_NR99_tax_silva.fasta.gz ] ||
  curl -fL -o SILVA_138.2_SSURef_NR99_tax_silva.fasta.gz \
    https://www.arb-silva.de/fileadmin/silva_databases/release_138.2/Exports/SILVA_138.2_SSURef_NR99_tax_silva.fasta.gz
[ -f pr2_version_5.1.1_SSU_taxo_long.fasta.gz ] ||
  curl -fL -o pr2_version_5.1.1_SSU_taxo_long.fasta.gz \
    https://github.com/pr2database/pr2database/releases/download/v5.1.1/pr2_version_5.1.1_SSU_taxo_long.fasta.gz
cd ..

common=(--min-members 10 --max-per-class 20 --holdout 50 --seed 42)
atlas evaluate --marker 16S --reference benchmark-data/SILVA_138.2_SSURef_NR99_tax_silva.fasta.gz "${common[@]}" \
  | tee docs/benchmark/16S-silva.json
atlas evaluate --marker 18S --reference benchmark-data/pr2_version_5.1.1_SSU_taxo_long.fasta.gz "${common[@]}" \
  | tee docs/benchmark/18S-pr2.json

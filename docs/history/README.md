# ATLAS v1 development history

These documents were written in September 2025, while ATLAS was built for
Smart India Hackathon problem statement SIH25042. They record how the four
marker pipelines were developed notebook-first, the decisions taken (k = 6,
at least three sequences per genus, Rishu Tiwari's lower learning rate and
longer early-stopping patience) and the problems solved along the way (GPU
setup for TensorFlow 2.10, VRAM exhaustion, stratification errors).

ATLAS v2 replaced the per-marker scripts with one package and command (see
the main [README](../../README.md)) and kept those decisions, except the
learning rate: on the full SILVA and PR2 references 1e-3 trains five times
faster than 1e-4 with slightly better accuracy (see
[BENCHMARK.md](../BENCHMARK.md#learning-rate)). The notebooks and scripts
these logs refer to are in the repository history.

- [Project overview (v1)](v1-project-overview.md)
- [16S data preparation workflow](v1-16S-workflow.md)
- Development logs: [16S](v1-dev-log-16S.md) · [18S](v1-dev-log-18S.md) · [COI](v1-dev-log-COI.md) · [ITS](v1-dev-log-ITS.md)

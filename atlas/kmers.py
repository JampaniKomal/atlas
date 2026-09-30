"""k-mer composition profiles.

A sequence becomes a vector of its k-mer counts over the alphabet ACGT, one
slot per possible k-mer (4**k slots: 4,096 for k=6). k-mers containing an
ambiguous base (N, R, Y, ...) are skipped. RNA input (U) is read as T, since
SILVA distributes its sequences as RNA.

Counting is vectorised with numpy: each base becomes a 2-bit code, and a
k-mer's slot is its base-4 number, built with k shifted additions over the
whole sequence at once. That replaces the per-k-mer Python dictionary of the
v1 scripts and is roughly two orders of magnitude faster.

Profiles are L2-normalised, so sequences of different lengths are comparable
and the dot product of two profiles is their cosine similarity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_CODES = np.full(256, 4, dtype=np.uint8)
for _i, _bases in enumerate(("Aa", "Cc", "Gg", "TtUu")):
    for _b in _bases:
        _CODES[ord(_b)] = _i


def n_features(k: int) -> int:
    return 4**k


def kmer_counts(seq: str, k: int) -> np.ndarray:
    """Counts of every k-mer in seq, as an int array of length 4**k."""
    out = np.zeros(4**k, dtype=np.int64)
    n = len(seq) - k + 1
    if n <= 0:
        return out
    codes = _CODES[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]
    index = np.zeros(n, dtype=np.int64)
    for j in range(k):
        index = index * 4 + np.minimum(codes[j : j + n], 3)
    ambiguous = codes >= 4
    if ambiguous.any():
        bad = np.convolve(ambiguous, np.ones(k, dtype=np.int64), mode="valid") > 0
        index = index[~bad]
    return np.bincount(index, minlength=4**k)


def profile(seq: str, k: int) -> np.ndarray:
    """L2-normalised k-mer profile (float32). All zeros if seq has no valid k-mer."""
    v = kmer_counts(seq, k).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def profiles(seqs: Sequence[str], k: int) -> np.ndarray:
    """Profiles for many sequences, stacked into an (n, 4**k) float32 matrix."""
    out = np.zeros((len(seqs), 4**k), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i] = profile(s, k)
    return out


def kmer_label(index: int, k: int) -> str:
    """The k-mer for a profile slot: kmer_label(27, 3) == "CGT" (27 = 1*16 + 2*4 + 3)."""
    bases = []
    for _ in range(k):
        index, r = divmod(index, 4)
        bases.append("ACGT"[r])
    return "".join(reversed(bases))

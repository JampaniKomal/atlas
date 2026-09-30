"""Turning a reference database into a labelled training set."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import references
from .fasta import Record, read_fasta
from .markers import Marker


@dataclass
class Dataset:
    ids: list[str]
    seqs: list[str]
    labels: list[str]
    # Higher ranks for each label (genus -> family, order, ...), for reports.
    lineages: dict[str, dict[str, str]] = field(default_factory=dict)
    source: str = ""
    sha256: str = ""

    def __len__(self) -> int:
        return len(self.labels)

    def classes(self) -> list[str]:
        return sorted(set(self.labels))

    def subset(self, keep: list[int]) -> Dataset:
        return Dataset(
            [self.ids[i] for i in keep],
            [self.seqs[i] for i in keep],
            [self.labels[i] for i in keep],
            self.lineages,
            self.source,
            self.sha256,
        )


def load_reference(
    path: str | Path,
    marker: Marker,
    *,
    rank: str = "genus",
    parser: str | None = None,
    min_members: int = 3,
    max_per_class: int | None = None,
    min_length: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Read a reference FASTA and keep sequences labelled at `rank`.

    Classes with fewer than `min_members` sequences are dropped: a genus seen
    once or twice can only be memorised, and it breaks stratified splitting
    (the v1 16S log records the ValueError that led to this threshold).
    `max_per_class` caps over-represented genera so they don't dominate.
    """
    parse = references.PARSERS[parser or marker.parser]
    min_len = marker.min_length if min_length is None else min_length
    by_label: dict[str, list[Record]] = defaultdict(list)
    lineages: dict[str, dict[str, str]] = {}
    digest = hashlib.sha256()
    for rec in read_fasta(path):
        lineage = parse(rec.description)
        if marker.domains and lineage.get("domain") not in marker.domains:
            continue
        label = lineage.get(rank)
        if not label or len(rec.seq) < min_len:
            continue
        by_label[label].append(rec)
        lineages.setdefault(label, {r: n for r, n in lineage.items() if r not in (rank, "species")})
        digest.update(rec.id.encode())

    rng = random.Random(seed)
    ids, seqs, labels = [], [], []
    for label in sorted(by_label):
        recs = by_label[label]
        if len(recs) < min_members:
            continue
        if max_per_class and len(recs) > max_per_class:
            recs = rng.sample(recs, max_per_class)
        for r in recs:
            ids.append(r.id)
            seqs.append(r.seq)
            labels.append(label)
    kept = set(labels)
    return Dataset(ids, seqs, labels, {k: v for k, v in lineages.items() if k in kept}, str(path), digest.hexdigest())


def stratified_split(labels: list[str], fractions: tuple[float, ...], seed: int = 42) -> list[list[int]]:
    """Split indices so every class appears in every part where it can.

    Each class is shuffled and divided by `fractions`; a class too small for
    all parts fills them in order, so the first (training) part always gets
    at least one example.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_class[label].append(i)
    parts: list[list[int]] = [[] for _ in fractions]
    for label in sorted(by_class):
        idx = by_class[label]
        rng.shuffle(idx)
        n = len(idx)
        counts = [max(1, round(f * n)) for f in fractions[1:]]
        while sum(counts) >= n and any(counts):
            counts[counts.index(max(counts))] -= 1
        start = n - sum(counts)
        parts[0].extend(idx[:start])
        for p, c in enumerate(counts, start=1):
            parts[p].extend(idx[start : start + c])
            start += c
    return parts


def class_summary(labels: list[str]) -> dict[str, int]:
    counts = Counter(labels)
    sizes = np.array(list(counts.values())) if counts else np.array([0])
    return {
        "sequences": len(labels),
        "classes": len(counts),
        "min_per_class": int(sizes.min()),
        "median_per_class": int(np.median(sizes)),
        "max_per_class": int(sizes.max()),
    }

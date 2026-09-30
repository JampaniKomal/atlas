"""The full analysis: route reads to a marker, classify (Filter), cluster the
rest (Explorer), and estimate abundance and diversity."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import kmers
from .diversity import alpha_diversity
from .explorer import Cluster, explore
from .fasta import Record
from .model import Classifier


@dataclass
class Assignment:
    id: str
    marker: str
    taxon: str | None
    confidence: float


@dataclass
class Result:
    sample: str
    sequences: int
    markers: dict[str, int]
    assignments: list[Assignment]
    abundance: dict[str, int]  # classified taxon -> reads, most abundant first
    lineages: dict[str, dict[str, str]]
    clusters: list[Cluster]
    noise: list[str]
    skipped: list[str]  # too short or no valid k-mers
    diversity: dict[str, float]
    seconds: float
    device: str
    models: dict[str, dict] = field(default_factory=dict)

    @property
    def classified(self) -> int:
        return sum(self.abundance.values())

    @property
    def unclassified(self) -> int:
        return self.sequences - self.classified - len(self.skipped)

    def chart_data(self) -> dict[str, int]:
        """Taxon -> reads, including Explorer clusters, for the web UI charts."""
        data = dict(self.abundance)
        for c in self.clusters:
            data[f"{c.id} (near {c.nearest})"] = c.size
        return data


def load_models(models_dir: str | Path, only: str | None = None) -> list[Classifier]:
    root = Path(models_dir)
    dirs = sorted(p.parent for p in root.glob("*/meta.json"))
    models = [Classifier(d) for d in dirs]
    if only and only.lower() != "auto":
        models = [m for m in models if m.name.upper() == only.upper()]
    return models


def device() -> str:
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        return f"GPU ({len(gpus)} found)" if gpus else "CPU"
    except Exception:  # pragma: no cover - TensorFlow import problems
        return "CPU"


def analyze(
    records: list[Record],
    classifiers: list[Classifier],
    *,
    sample: str = "sample",
    min_cluster_size: int = 5,
    min_length: int = 50,
) -> Result:
    if not classifiers:
        raise ValueError("no trained models found (train one with `atlas train`, or try `atlas demo`)")
    started = time.time()
    skipped = [r.id for r in records if len(r.seq) < min_length]
    usable = [r for r in records if len(r.seq) >= min_length]

    # Route each read to the marker whose known classes it most resembles.
    # Different k per marker means profiles are computed per marker.
    profiles = {c.name: kmers.profiles([r.seq for r in usable], c.meta.k) for c in classifiers}
    empty = np.array([not np.any(profiles[classifiers[0].name][i]) for i in range(len(usable))], dtype=bool)
    skipped += [r.id for r, e in zip(usable, empty, strict=True) if e]
    if len(classifiers) == 1:
        route = np.zeros(len(usable), dtype=int)
    else:
        affinity = np.stack([(profiles[c.name] @ c.centroids.T).max(axis=1) for c in classifiers], axis=1)
        route = affinity.argmax(axis=1)

    assignments: list[Assignment] = []
    clusters: list[Cluster] = []
    noise: list[str] = []
    abundance: Counter[str] = Counter()
    lineages: dict[str, dict[str, str]] = {}
    marker_counts: dict[str, int] = {}
    for m, clf in enumerate(classifiers):
        idx = [i for i in range(len(usable)) if route[i] == m and not empty[i]]
        marker_counts[clf.name] = len(idx)
        if not idx:
            continue
        X = profiles[clf.name][idx]
        labels, conf = clf.classify(X)
        unknown = []
        for j, (label, c) in enumerate(zip(labels, conf, strict=True)):
            rec = usable[idx[j]]
            assignments.append(Assignment(rec.id, clf.name, label, round(float(c), 4)))
            if label is None:
                unknown.append(j)
            else:
                abundance[label] += 1
                lineages[label] = clf.meta.lineages.get(label, {})
        if unknown:
            found, left = explore(
                X[unknown],
                [usable[idx[j]].id for j in unknown],
                [usable[idx[j]].seq for j in unknown],
                marker=clf.name,
                centroids=clf.centroids,
                labels=clf.meta.labels,
                lineages=clf.meta.lineages,
                novelty_similarity=clf.meta.novelty_similarity,
                min_cluster_size=min_cluster_size,
            )
            clusters += found
            noise += left

    units = Counter(abundance)
    for c in clusters:
        units[c.id] = c.size
    return Result(
        sample=sample,
        sequences=len(records),
        markers=marker_counts,
        assignments=assignments,
        abundance=dict(abundance.most_common()),
        lineages=lineages,
        clusters=clusters,
        noise=noise,
        skipped=skipped,
        diversity=alpha_diversity(units),
        seconds=round(time.time() - started, 2),
        device=device(),
        models={
            c.name: {
                "rank": c.meta.rank,
                "k": c.meta.k,
                "classes": len(c.meta.labels),
                "threshold": c.meta.threshold,
                "trained_on": c.meta.data.get("source", ""),
            }
            for c in classifiers
        },
    )

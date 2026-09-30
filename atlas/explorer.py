"""The Explorer: unsupervised discovery among reads the Filter couldn't classify.

Reads that no classifier is confident about are the interesting ones in a
deep-sea sample: they may belong to lineages missing from the reference
databases. The Explorer groups them by k-mer composition (PCA, then HDBSCAN,
which finds clusters of varying density and leaves outliers as noise) and
describes each cluster by its nearest known taxon.

A cluster is flagged as a putative novel lineage when its centroid is less
similar to every known class centroid than 95% of the training sequences are
to their own class centroid. That is a statement about distance from the
reference, not a species description: representative sequences are reported
so they can be checked with BLAST or placed on a phylogeny.

v1 embedded sequences with a Doc2Vec model trained on the reference; v2
clusters the k-mer profiles directly, which needs no second model and is
deterministic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Cluster:
    id: str
    marker: str
    size: int
    members: list[str]
    representative: str
    representative_seq: str
    nearest: str
    nearest_lineage: dict[str, str]
    similarity: float
    novel: bool

    def to_dict(self) -> dict:
        return asdict(self)


def explore(
    X: np.ndarray,
    ids: list[str],
    seqs: list[str],
    *,
    marker: str,
    centroids: np.ndarray,
    labels: list[str],
    lineages: dict[str, dict[str, str]],
    novelty_similarity: float,
    min_cluster_size: int = 5,
    seed: int = 0,
) -> tuple[list[Cluster], list[str]]:
    """Cluster unclassified reads. Returns the clusters and the IDs left as noise."""
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA

    n = len(X)
    if n < max(2, min_cluster_size):
        return [], list(ids)
    Z = X
    components = min(50, n - 1, X.shape[1])
    if components >= 2:
        Z = PCA(n_components=components, random_state=seed).fit_transform(X)
    assignment = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, cluster_selection_method="eom", copy=True).fit_predict(
        Z
    )

    clusters: list[Cluster] = []
    for c in sorted(set(assignment) - {-1}, key=lambda c: -int(np.sum(assignment == c))):
        members = np.nonzero(assignment == c)[0]
        centroid = X[members].mean(axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        rep = members[int(np.argmax(X[members] @ centroid))]
        sims = centroids @ centroid
        best = int(np.argmax(sims))
        clusters.append(
            Cluster(
                id=f"{marker}-X{len(clusters) + 1}",
                marker=marker,
                size=len(members),
                members=[ids[i] for i in members],
                representative=ids[rep],
                representative_seq=seqs[rep],
                nearest=labels[best],
                nearest_lineage=lineages.get(labels[best], {}),
                similarity=round(float(sims[best]), 4),
                novel=bool(sims[best] < novelty_similarity),
            )
        )
    noise = [ids[i] for i in np.nonzero(assignment == -1)[0]]
    return clusters, noise

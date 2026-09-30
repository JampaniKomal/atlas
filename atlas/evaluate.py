"""The held-out-genera benchmark: how ATLAS behaves on taxa missing from its
reference, which is the situation it was built for.

Whole genera are removed before training. The model is then tested on:
  - held-out sequences of the genera it knows (closed-set accuracy, and
    precision and coverage at the calibrated confidence threshold);
  - every sequence of the removed genera ("novel"): how many the Filter
    correctly refuses to classify, and whether the Explorer groups them by
    their true genus (adjusted Rand index, homogeneity) and flags them.
"""

from __future__ import annotations

import random
import tempfile
from collections import Counter

import numpy as np

from . import kmers
from .dataset import Dataset, stratified_split
from .explorer import explore
from .model import Classifier, TrainConfig, train


def novelty_benchmark(
    data: Dataset,
    marker: str,
    rank: str,
    *,
    holdout: int = 10,
    cfg: TrainConfig | None = None,
    min_cluster_size: int = 5,
    seed: int = 42,
    log=print,
) -> dict:
    from sklearn.metrics import adjusted_rand_score, homogeneity_score, roc_auc_score

    cfg = cfg or TrainConfig(seed=seed)
    counts = Counter(data.labels)
    eligible = sorted(c for c, n in counts.items() if n >= min_cluster_size)
    rng = random.Random(seed)
    held = set(rng.sample(eligible, min(holdout, len(eligible) // 5 or 1)))
    known = data.subset([i for i, label in enumerate(data.labels) if label not in held])
    novel = data.subset([i for i, label in enumerate(data.labels) if label in held])
    log(f"benchmark: {len(known.classes())} known {rank} classes, {len(held)} held out ({len(novel)} sequences)")

    with tempfile.TemporaryDirectory() as tmp:
        meta = train(known, marker, rank, tmp, cfg, log=log)
        clf = Classifier(tmp)
        _, _, test_idx = stratified_split(known.labels, (0.7, 0.15, 0.15), seed=cfg.seed)
        X_known = kmers.profiles([known.seqs[i] for i in test_idx], meta.k)
        X_novel = kmers.profiles(novel.seqs, meta.k)
        known_labels, known_conf = clf.classify(X_known)
        novel_labels, novel_conf = clf.classify(X_novel)
        known_top = clf.predict_proba(X_known).argmax(axis=1)

        rejected = [i for i, label in enumerate(novel_labels) if label is None]
        clusters, noise = explore(
            X_novel[rejected],
            [novel.ids[i] for i in rejected],
            [novel.seqs[i] for i in rejected],
            marker=marker,
            centroids=clf.centroids,
            labels=clf.meta.labels,
            lineages=clf.meta.lineages,
            novelty_similarity=clf.meta.novelty_similarity,
            min_cluster_size=min_cluster_size,
        )

    truth = {novel.ids[i]: novel.labels[i] for i in rejected}
    assigned = {m: c.id for c in clusters for m in c.members}
    clustered = [sid for sid in truth if sid in assigned]
    ari = adjusted_rand_score([truth[s] for s in clustered], [assigned[s] for s in clustered]) if clustered else 0.0
    homogeneity = homogeneity_score([truth[s] for s in clustered], [assigned[s] for s in clustered]) if clustered else 0.0
    recovered = set()
    for c in clusters:
        majority, n = Counter(truth[m] for m in c.members).most_common(1)[0]
        if n / c.size >= 0.8:
            recovered.add(majority)

    y_known = np.array([known.labels[i] for i in test_idx])
    # Open-set view: how well the classifier's confidence separates known from
    # novel sequences, and the trade-off at other thresholds than the calibrated one.
    index = {label: i for i, label in enumerate(clf.meta.labels)}
    y_idx = np.array([index[label] for label in y_known])
    sweep = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        keep = known_conf >= t
        sweep.append(
            {
                "threshold": t,
                "known_coverage": round(float(keep.mean()), 4),
                "known_precision": round(float((known_top[keep] == y_idx[keep]).mean()), 4) if keep.any() else 0.0,
                "novel_rejected": round(float((novel_conf < t).mean()), 4),
            }
        )
    auroc = roc_auc_score(np.r_[np.ones(len(known_conf)), np.zeros(len(novel_conf))], np.r_[known_conf, novel_conf])
    accepted = np.array([label is not None for label in known_labels])
    correct = np.array([p == t for p, t in zip(known_labels, y_known, strict=True)])
    result = {
        "known_classes": len(known.classes()),
        "held_out_classes": len(held),
        "known_test_sequences": len(test_idx),
        "novel_sequences": len(novel),
        "closed_set_accuracy": meta.metrics["test_accuracy"],
        "closed_set_macro_f1": meta.metrics["test_macro_f1"],
        "threshold": meta.threshold,
        "known_coverage": float(accepted.mean()) if len(accepted) else 0.0,
        "known_precision": float(correct[accepted].mean()) if accepted.any() else 0.0,
        "novel_rejected": len(rejected) / max(len(novel), 1),
        "novel_clustered": len(clustered) / max(len(rejected), 1),
        "explorer_clusters": len(clusters),
        "explorer_ari": round(float(ari), 4),
        "explorer_homogeneity": round(float(homogeneity), 4),
        "held_out_recovered": len(recovered) / max(len(held), 1),
        "clusters_flagged_novel": sum(c.novel for c in clusters) / max(len(clusters), 1),
        "unclustered": len(noise),
        "novelty_auroc": round(float(auroc), 4),
        "epochs": meta.metrics["epochs"],
        "train_seconds": meta.metrics["train_seconds"],
        "threshold_sweep": sweep,
    }
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}

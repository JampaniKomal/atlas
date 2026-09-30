"""The Filter: one deep-learning classifier per marker gene.

A multilayer perceptron over k-mer profiles, as in ATLAS v1. Training keeps
the longer early-stopping patience Rishu Tiwari gave the v1 pipelines (5
epochs instead of 3, restoring the best weights). He also lowered v1's Adam
learning rate to 1e-4; on full SILVA and PR2 references 1e-3 converges
5x faster with slightly better accuracy (docs/BENCHMARK.md), so it is the
default, and --learning-rate 1e-4 is still available.

A trained marker lives in one directory:

    model.keras     the network
    meta.json       labels, lineages, k, rank, threshold, metrics, provenance
    centroids.npy   mean profile of each class (for routing and the Explorer)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import __version__, kmers
from .dataset import Dataset, class_summary, stratified_split

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@dataclass
class TrainConfig:
    k: int = 6
    hidden: tuple[int, ...] = (1024, 512)
    dropout: float = 0.5
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 5
    target_precision: float = 0.95
    seed: int = 42


@dataclass
class ModelMeta:
    marker: str
    rank: str
    k: int
    labels: list[str]
    lineages: dict[str, dict[str, str]]
    threshold: float
    novelty_similarity: float
    metrics: dict[str, float] = field(default_factory=dict)
    data: dict[str, object] = field(default_factory=dict)
    config: dict[str, object] = field(default_factory=dict)
    atlas_version: str = __version__
    trained_at: str = ""


class Classifier:
    """A trained marker classifier loaded from its directory."""

    def __init__(self, directory: str | Path):
        import keras

        self.dir = Path(directory)
        self.meta = ModelMeta(**json.loads((self.dir / "meta.json").read_text()))
        self.model = keras.models.load_model(self.dir / "model.keras")
        self.centroids = np.load(self.dir / "centroids.npy")

    @property
    def name(self) -> str:
        return self.meta.marker

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, batch_size=256, verbose=0)

    def classify(self, X: np.ndarray) -> tuple[list[str | None], np.ndarray]:
        """Labels (None below the confidence threshold) and top probabilities."""
        proba = self.predict_proba(X)
        top = proba.argmax(axis=1)
        conf = proba[np.arange(len(top)), top]
        labels = [self.meta.labels[t] if c >= self.meta.threshold else None for t, c in zip(top, conf, strict=True)]
        return labels, conf


def _build(n_features: int, n_classes: int, cfg: TrainConfig):
    import keras

    keras.utils.set_random_seed(cfg.seed)
    layers = [keras.Input(shape=(n_features,))]
    for units in cfg.hidden:
        layers += [keras.layers.Dense(units, activation="relu"), keras.layers.Dropout(cfg.dropout)]
    layers.append(keras.layers.Dense(n_classes, activation="softmax"))
    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def calibrate_threshold(conf: np.ndarray, correct: np.ndarray, target_precision: float) -> float:
    """Lowest confidence cut-off at which accepted predictions reach the target
    precision on held-out data. Falls back to 0.8 (the v1 value) if none does."""
    order = np.argsort(-conf)
    hits = np.cumsum(correct[order])
    precision = hits / np.arange(1, len(order) + 1)
    ok = np.nonzero(precision >= target_precision)[0]
    if len(ok) == 0:
        return 0.8
    return float(conf[order][ok[-1]])


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def class_centroids(X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Unit-length mean profile of each class, in one sparse matrix product."""
    from scipy.sparse import csr_matrix

    onehot = csr_matrix((np.ones(len(y), dtype=np.float32), (y, np.arange(len(y)))), shape=(n_classes, len(y)))
    sums = np.asarray(onehot @ X, dtype=np.float32)
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    return sums / np.where(norms > 0, norms, 1.0)


def train(
    data: Dataset,
    marker: str,
    rank: str,
    out_dir: str | Path,
    cfg: TrainConfig | None = None,
    log=print,
) -> ModelMeta:
    """Train, calibrate, evaluate on a held-out test split, and save."""
    import keras

    cfg = cfg or TrainConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = data.classes()
    index = {label: i for i, label in enumerate(labels)}
    y_all = np.array([index[label] for label in data.labels])

    log(
        f"{marker}: {len(data)} sequences, {len(labels)} {rank} classes; computing {kmers.n_features(cfg.k)} {cfg.k}-mer features"
    )
    X_all = kmers.profiles(data.seqs, cfg.k)
    train_idx, val_idx, test_idx = stratified_split(data.labels, (0.7, 0.15, 0.15), seed=cfg.seed)

    model = _build(X_all.shape[1], len(labels), cfg)
    started = time.time()
    history = model.fit(
        X_all[train_idx],
        y_all[train_idx],
        validation_data=(X_all[val_idx], y_all[val_idx]),
        epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True),
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: log(
                    f"{marker}: epoch {epoch + 1}: loss {logs['loss']:.3f}, validation accuracy {logs['val_accuracy']:.3f}"
                )
            ),
        ],
        verbose=0,
    )
    seconds = time.time() - started

    def evaluate(idx):
        proba = model.predict(X_all[idx], batch_size=256, verbose=0)
        pred = proba.argmax(axis=1)
        return pred, proba[np.arange(len(pred)), pred]

    val_pred, val_conf = evaluate(val_idx)
    threshold = calibrate_threshold(val_conf, val_pred == y_all[val_idx], cfg.target_precision)
    test_pred, test_conf = evaluate(test_idx)
    y_test = y_all[test_idx]
    accepted = test_conf >= threshold
    metrics = {
        "test_accuracy": float(np.mean(test_pred == y_test)),
        "test_macro_f1": macro_f1(y_test, test_pred, len(labels)),
        "test_coverage_at_threshold": float(np.mean(accepted)),
        "test_precision_at_threshold": float(np.mean(test_pred[accepted] == y_test[accepted])) if accepted.any() else 0.0,
        "epochs": len(history.history["loss"]),
        "train_seconds": round(seconds, 1),
    }

    centroids = class_centroids(X_all[train_idx], y_all[train_idx], len(labels))
    # How far known sequences sit from their own class centroid: the Explorer
    # calls a cluster novel when it is farther than 95% of these.
    own = np.einsum("ij,ij->i", X_all[train_idx], centroids[y_all[train_idx]])
    novelty_similarity = float(np.quantile(own, 0.05))

    meta = ModelMeta(
        marker=marker,
        rank=rank,
        k=cfg.k,
        labels=labels,
        lineages={label: data.lineages.get(label, {}) for label in labels},
        threshold=round(threshold, 4),
        novelty_similarity=round(novelty_similarity, 4),
        metrics=metrics,
        data={
            **class_summary(data.labels),
            "source": Path(data.source).name,
            "sha256": data.sha256,
            "split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        },
        config={k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
        trained_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    model.save(out / "model.keras")
    np.save(out / "centroids.npy", centroids)
    (out / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
    log(
        f"{marker}: test accuracy {metrics['test_accuracy']:.3f}, macro-F1 {metrics['test_macro_f1']:.3f}; "
        f"threshold {threshold:.2f} keeps {metrics['test_coverage_at_threshold']:.1%} at "
        f"{metrics['test_precision_at_threshold']:.1%} precision ({metrics['epochs']} epochs, {seconds:.0f}s)"
    )
    return meta

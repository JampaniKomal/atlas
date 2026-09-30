"""The `atlas` command."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from . import __version__

# TensorFlow prints several lines of C++ start-up logging (oneDNN notices, "no
# CUDA driver" on CPU-only machines) that say nothing about the analysis.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

BANNER = r"""
    _  _____ _        _    ____
   / \|_   _| |      / \  / ___|
  / _ \ | | | |     / _ \ \___ \
 / ___ \| | | |___ / ___ \ ___) |
/_/   \_\_| |_____/_/   \_\____/
 Artificial Taxonomic Learning & Analysis System
"""


def _log(quiet: bool):
    return (lambda *a, **k: None) if quiet else (lambda msg: print(msg, file=sys.stderr))


def cmd_train(a: argparse.Namespace) -> int:
    from .dataset import load_reference
    from .markers import get
    from .model import TrainConfig, train

    marker = get(a.marker)
    log = _log(a.quiet)
    log(f"reading {a.reference} ({marker.description})")
    data = load_reference(
        a.reference,
        marker,
        rank=a.rank,
        parser=a.parser,
        min_members=a.min_members,
        max_per_class=a.max_per_class,
        min_length=a.min_length,
    )
    if len(data.classes()) < 2:
        print("atlas: fewer than two classes left after filtering; check --parser, --rank and --min-members", file=sys.stderr)
        return 2
    cfg = TrainConfig(
        k=a.k or marker.k,
        learning_rate=a.learning_rate,
        max_epochs=a.epochs,
        batch_size=a.batch_size,
        hidden=tuple(a.hidden),
        target_precision=a.target_precision,
    )
    train(data, marker.name, a.rank, Path(a.models) / marker.name, cfg, log=log)
    return 0


def _read_input(path: Path | None):
    from .fasta import read_fasta

    if path is None:
        print(BANNER)
        print("Enter the path to a FASTA file (for example data/sample.fasta).")
        path = Path(input("FASTA file: ").strip().strip('"'))
    if not path.is_file():
        print(f"atlas: no such file: {path}", file=sys.stderr)
        return None, path
    return list(read_fasta(path)), path


def cmd_analyze(a: argparse.Namespace) -> int:
    from .analysis import analyze, load_models
    from .report import to_html, to_json, to_text

    records, path = _read_input(a.fasta)
    if records is None:
        return 2
    models = load_models(a.models, a.marker)
    if not models:
        print(f"atlas: no trained models in {a.models} (run `atlas train`, or `atlas demo` to try it)", file=sys.stderr)
        return 2
    if a.threshold is not None:
        if not 0 <= a.threshold <= 1:
            print("atlas: --threshold must be between 0 and 1", file=sys.stderr)
            return 2
        for m in models:
            m.meta.threshold = a.threshold
    _log(a.quiet)(f"analysing {len(records)} sequences with {', '.join(m.name for m in models)}")
    result = analyze(records, models, sample=path.name, min_cluster_size=a.min_cluster_size)
    text = to_text(result)
    print(text)
    reports = Path(a.reports)
    reports.mkdir(parents=True, exist_ok=True)
    name = a.report_name or f"ATLAS_REPORT_{path.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    (reports / f"{name}.txt").write_text(text, encoding="utf-8")
    written = [reports / f"{name}.txt"]
    if a.html is not False:
        target = Path(a.html) if a.html else reports / f"{name}.html"
        target.write_text(to_html(result), encoding="utf-8")
        written.append(target)
    if a.json:
        Path(a.json).write_text(to_json(result), encoding="utf-8")
        written.append(Path(a.json))
    _log(a.quiet)("reports: " + ", ".join(str(p) for p in written))
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .server import create_app

    app = create_app(a.models, a.min_cluster_size)
    print(BANNER)
    print(f"ATLAS web interface on http://{a.host}:{a.port}  (models: {a.models})")
    app.run(host=a.host, port=a.port, debug=False)
    return 0


def cmd_info(a: argparse.Namespace) -> int:
    rows = sorted(Path(a.models).glob("*/meta.json"))
    if not rows:
        print(f"no trained models in {a.models}")
        return 1
    for meta_path in rows:
        m = json.loads(meta_path.read_text())
        met = m["metrics"]
        print(
            f"{m['marker']:<4} {len(m['labels'])} {m['rank']} classes, k={m['k']}, trained {m['trained_at']} on "
            f"{m['data'].get('source')} ({m['data'].get('sequences')} seqs); test accuracy {met['test_accuracy']:.3f}, "
            f"macro-F1 {met['test_macro_f1']:.3f}, threshold {m['threshold']}"
        )
    return 0


def cmd_demo(a: argparse.Namespace) -> int:
    from .analysis import analyze, load_models
    from .dataset import load_reference
    from .fasta import write_fasta
    from .markers import get
    from .model import TrainConfig, train
    from .report import to_html, to_text
    from .synthetic import make_reference, make_sample, make_tree

    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    log = _log(a.quiet)
    known = make_tree(4, 3, 3, length=1200, rng=rng)
    novel = make_tree(1, 2, 2, length=1200, rng=rng, prefix="Nova")
    write_fasta(make_reference(known, 6, rng), out / "reference.fasta")
    reads, truth = make_sample(known, novel, a.reads, rng)
    write_fasta(reads, out / "sample.fasta")
    log(
        f"synthetic reference: {len(known)} genera; sample: {len(reads)} reads, "
        f"{sum(v.startswith('NOVEL') for v in truth.values())} from {len(novel)} genera absent from the reference"
    )

    data = load_reference(out / "reference.fasta", get("16S"), min_length=100)
    cfg = TrainConfig(hidden=(256, 128), learning_rate=1e-3, max_epochs=80, patience=8, seed=a.seed)
    train(data, "16S", "genus", out / "models" / "16S", cfg, log=log)
    result = analyze(reads, load_models(out / "models"), sample="sample.fasta")
    (out / "report.txt").write_text(to_text(result), encoding="utf-8")
    (out / "report.html").write_text(to_html(result), encoding="utf-8")
    print(to_text(result))

    by_read = {x.id: x.taxon for x in result.assignments}
    known_reads = [r for r, t in truth.items() if not t.startswith("NOVEL")]
    novel_reads = [r for r, t in truth.items() if t.startswith("NOVEL")]
    correct = sum(by_read.get(r) == truth[r] for r in known_reads)
    refused = sum(by_read.get(r) is None for r in novel_reads)
    flagged = sum(c.size for c in result.clusters if c.novel)
    print(
        f"\nAgainst the known truth: {correct}/{len(known_reads)} known reads classified correctly; "
        f"{refused}/{len(novel_reads)} reads from absent genera left unclassified; "
        f"{flagged} reads in clusters flagged as putative novel lineages."
    )
    print(f"Files in {out}: reference.fasta, sample.fasta, models/, report.txt, report.html")
    print(f"Next: atlas serve --models {out / 'models'}")
    return 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    from .dataset import load_reference
    from .evaluate import novelty_benchmark
    from .markers import get
    from .model import TrainConfig

    marker = get(a.marker)
    data = load_reference(
        a.reference,
        marker,
        rank=a.rank,
        parser=a.parser,
        min_members=a.min_members,
        max_per_class=a.max_per_class,
        min_length=a.min_length,
    )
    cfg = TrainConfig(
        k=a.k or marker.k,
        learning_rate=a.learning_rate,
        max_epochs=a.epochs,
        batch_size=a.batch_size,
        hidden=tuple(a.hidden),
        seed=a.seed,
    )
    result = novelty_benchmark(
        data, marker.name, a.rank, holdout=a.holdout, cfg=cfg, min_cluster_size=a.min_cluster_size, seed=a.seed, log=_log(a.quiet)
    )
    print(json.dumps(result, indent=2))
    return 0


def _reference_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--marker", required=True, help="16S, 18S, COI or ITS")
    p.add_argument("--reference", required=True, type=Path, help="reference FASTA (plain or .gz)")
    p.add_argument(
        "--parser", choices=["silva", "pr2", "unite", "midori", "lineage"], help="header format (default: the marker's)"
    )
    p.add_argument("--rank", default="genus", help="taxonomic rank to classify at (default genus)")
    p.add_argument("--min-members", type=int, default=3, help="drop classes with fewer sequences (default 3)")
    p.add_argument("--max-per-class", type=int, help="cap sequences per class")
    p.add_argument("--min-length", type=int, help="drop shorter sequences (default: the marker's)")
    p.add_argument("--k", type=int, help="k-mer size (default: the marker's, 6)")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 512], help="hidden layer sizes")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate (default 1e-3)")
    p.add_argument("--epochs", type=int, default=100, help="maximum epochs (early stopping ends sooner)")
    p.add_argument("--batch-size", type=int, default=64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atlas", description="ATLAS: taxonomy and biodiversity from environmental DNA.")
    p.add_argument("--version", action="version", version=f"atlas {__version__}")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress messages")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a marker classifier from a reference database")
    _reference_args(t)
    t.add_argument("--models", default="models", help="where trained models live (default ./models)")
    t.add_argument(
        "--target-precision",
        type=float,
        default=0.95,
        help="calibrate the confidence threshold to this precision on validation data",
    )
    t.set_defaults(func=cmd_train)

    an = sub.add_parser("analyze", aliases=["analyse"], help="classify, cluster and summarise a FASTA sample")
    an.add_argument("fasta", nargs="?", type=Path, help="input FASTA (prompts if omitted)")
    an.add_argument("--models", default="models")
    an.add_argument("--marker", default="auto", help="use only this marker's model (default: route automatically)")
    an.add_argument(
        "--threshold",
        type=float,
        help="confidence cut-off for every model instead of its calibrated one; "
        "higher sends more reads to the Explorer (see docs/BENCHMARK.md)",
    )
    an.add_argument("--min-cluster-size", type=int, default=5, help="smallest Explorer cluster (default 5)")
    an.add_argument("--reports", default="reports", help="report directory (default ./reports)")
    an.add_argument("--report-name", help="report file name without extension")
    an.add_argument("--html", nargs="?", const="", default="", help="HTML report path (written by default)")
    an.add_argument("--no-html", dest="html", action="store_false", help="skip the HTML report")
    an.add_argument("--json", help="also write a JSON report here")
    an.set_defaults(func=cmd_analyze)

    s = sub.add_parser("serve", help="run the web interface")
    s.add_argument("--models", default="models")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5000)
    s.add_argument("--min-cluster-size", type=int, default=5)
    s.set_defaults(func=cmd_serve)

    i = sub.add_parser("info", help="list trained models and their test metrics")
    i.add_argument("--models", default="models")
    i.set_defaults(func=cmd_info)

    d = sub.add_parser("demo", help="train and analyse on a synthetic community, end to end, in about a minute")
    d.add_argument("--dir", default="atlas-demo")
    d.add_argument("--reads", type=int, default=600)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("evaluate", help="held-out-genera benchmark on a reference database")
    _reference_args(e)
    e.add_argument("--holdout", type=int, default=10, help="number of genera to remove before training")
    e.add_argument("--min-cluster-size", type=int, default=5)
    e.add_argument("--seed", type=int, default=42)
    e.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except (KeyError, FileNotFoundError, ValueError) as e:
        print(f"atlas: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

"""End to end on a synthetic community: train, classify, explore, report, serve."""

import json
import random

import pytest

from atlas import report
from atlas.analysis import analyze, load_models
from atlas.cli import main
from atlas.dataset import load_reference
from atlas.fasta import write_fasta
from atlas.markers import get
from atlas.model import TrainConfig, train
from atlas.synthetic import make_reference, make_sample, make_tree


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("atlas")
    rng = random.Random(3)
    known = make_tree(3, 3, 2, length=900, rng=rng)
    novel = make_tree(1, 2, 2, length=900, rng=rng, prefix="Nova")
    write_fasta(make_reference(known, 6, rng), tmp / "ref.fasta")
    reads, truth = make_sample(known, novel, 300, rng, novel_fraction=0.2)
    write_fasta(reads, tmp / "sample.fasta")
    data = load_reference(tmp / "ref.fasta", get("16S"), min_length=100)
    cfg = TrainConfig(hidden=(128,), learning_rate=2e-3, max_epochs=60, patience=6, seed=3)
    meta = train(data, "16S", "genus", tmp / "models" / "16S", cfg, log=lambda *_: None)
    return tmp, reads, truth, meta


def test_training_learns_the_known_genera(trained):
    _, _, _, meta = trained
    assert len(meta.labels) == 9
    assert meta.metrics["test_accuracy"] >= 0.9
    assert 0 < meta.threshold <= 1 and 0 < meta.novelty_similarity < 1


def test_filter_classifies_known_and_explorer_finds_novel(trained):
    tmp, reads, truth, _ = trained
    result = analyze(reads, load_models(tmp / "models"), sample="sample.fasta")
    by_read = {a.id: a.taxon for a in result.assignments}
    known = [r for r, t in truth.items() if not t.startswith("NOVEL")]
    novel = [r for r, t in truth.items() if t.startswith("NOVEL")]
    accepted = [r for r in known if by_read[r] is not None]
    assert sum(by_read[r] == truth[r] for r in accepted) / len(accepted) >= 0.95
    assert sum(by_read[r] is None for r in novel) / len(novel) >= 0.8
    novel_clusters = [c for c in result.clusters if c.novel]
    assert novel_clusters, "no cluster was flagged as a putative novel lineage"
    for c in novel_clusters:
        members = [truth[m] for m in c.members]
        assert max(members.count(x) for x in set(members)) / len(members) >= 0.8
    assert result.diversity["observed"] == len(result.abundance) + len(result.clusters)

    text = report.to_text(result)
    assert "PART 2: EXPLORER RESULTS" in text and "PUTATIVE NOVEL LINEAGE" in text
    page = report.to_html(result)
    assert page.startswith("<!doctype html>") and "putative novel" in page
    data = json.loads(report.to_json(result))
    assert data["classified"] + data["unclassified"] + data["skipped"] == len(reads)


def test_cli_analyze_and_info(trained, tmp_path, capsys):
    tmp, _, _, _ = trained
    code = main(
        [
            "-q",
            "analyze",
            str(tmp / "sample.fasta"),
            "--models",
            str(tmp / "models"),
            "--reports",
            str(tmp_path),
            "--report-name",
            "r",
            "--json",
            str(tmp_path / "r.json"),
        ]
    )
    assert code == 0
    assert (tmp_path / "r.txt").exists() and (tmp_path / "r.html").exists() and (tmp_path / "r.json").exists()
    assert main(["info", "--models", str(tmp / "models")]) == 0
    assert "16S  9 genus classes" in capsys.readouterr().out
    assert main(["-q", "analyze", str(tmp / "missing.fasta"), "--models", str(tmp / "models")]) == 2


def test_cli_threshold_override(trained, tmp_path):
    tmp, _, _, _ = trained
    base = ["-q", "analyze", str(tmp / "sample.fasta"), "--models", str(tmp / "models"), "--reports", str(tmp_path)]
    assert main([*base, "--no-html", "--json", str(tmp_path / "all.json"), "--threshold", "0"]) == 0
    data = json.loads((tmp_path / "all.json").read_text(encoding="utf-8"))
    assert data["unclassified"] == 0
    assert main([*base, "--threshold", "1.5"]) == 2


def test_web_interface(trained):
    from atlas.server import create_app

    tmp, reads, _, _ = trained
    client = create_app(tmp / "models").test_client()
    assert b"/run_analysis" in client.get("/").data
    fasta = "".join(f">{r.id}\n{r.seq}\n" for r in reads[:60])
    res = client.post("/run_analysis", data={"data": fasta})
    body = res.get_json()
    assert res.status_code == 200 and body["status"] == "success"
    assert body["classified_results"] and "FINAL REPORT" in body["report_content"]
    assert client.get(body["html_report"]).status_code == 200
    assert client.post("/run_analysis", data={"data": ""}).status_code == 400

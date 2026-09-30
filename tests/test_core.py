import gzip
import math

import numpy as np
import pytest

from atlas import kmers, references
from atlas.dataset import stratified_split
from atlas.diversity import alpha_diversity
from atlas.fasta import parse_fasta, read_fasta
from atlas.model import calibrate_threshold, macro_f1


def test_kmer_counts_match_a_naive_count():
    seq = "ACGTTGCANNACGTACGU"
    for k in (1, 3, 4):
        expected = np.zeros(4**k, dtype=int)
        s = seq.replace("U", "T")
        for i in range(len(s) - k + 1):
            kmer = s[i : i + k]
            if set(kmer) <= set("ACGT"):
                expected[int("".join(str("ACGT".index(b)) for b in kmer), 4)] += 1
        assert (kmers.kmer_counts(seq, k) == expected).all(), k


def test_profile_is_unit_length_and_label_round_trips():
    v = kmers.profile("ACGT" * 50, 6)
    assert math.isclose(float(np.linalg.norm(v)), 1.0, rel_tol=1e-5)
    assert kmers.kmer_label(27, 3) == "CGT"
    assert not kmers.profile("NNNN", 3).any()
    assert not kmers.profile("AC", 6).any()


def test_fasta_parsing(tmp_path):
    text = ">a desc\nacgt\nACGT\n>b\nTTTT\n"
    recs = parse_fasta(text)
    assert [(r.id, r.seq) for r in recs] == [("a", "ACGTACGT"), ("b", "TTTT")]
    assert parse_fasta("ACGT ACGT")[0].seq == "ACGTACGT"
    gz = tmp_path / "x.fasta.gz"
    with gzip.open(gz, "wt") as f:
        f.write(text)
    assert [r.id for r in read_fasta(gz)] == ["a", "b"]


@pytest.mark.parametrize(
    "parser, header, genus, species, domain",
    [
        (
            "silva",
            "AB001.1.1500 Bacteria;Proteobacteria;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Escherichia-Shigella;Escherichia coli",
            "Escherichia-Shigella",
            "Escherichia coli",
            "Bacteria",
        ),
        (
            "silva",
            "AB002.1.1400 Bacteria;Firmicutes;Bacilli;Lactobacillales;uncultured;uncultured bacterium",
            None,
            None,
            "Bacteria",
        ),
        (
            "pr2",
            "AB000912.1.1709_U|Eukaryota|TSAR|Alveolata|Dinoflagellata|Dinophyceae|Peridiniales|Peridiniales_X|Heterocapsa|Heterocapsa_triquetra",
            "Heterocapsa",
            "Heterocapsa triquetra",
            "Eukaryota",
        ),
        (
            "pr2",
            "AB353770.1.1740_U|18S_rRNA|nucleus||Eukaryota|TSAR|Alveolata|Dinoflagellata|Dinophyceae|Peridiniales|Kryptoperidiniaceae|Unruhdinium|Unruhdinium_kevei",
            "Unruhdinium",
            "Unruhdinium kevei",
            "Eukaryota",
        ),
        (
            "unite",
            "Abrothallus_subhalei|MT153946|SH1227328.10FU|refs|k__Fungi;p__Ascomycota;c__Dothideomycetes;o__Abrothallales;f__Abrothallaceae;g__Abrothallus;s__Abrothallus_subhalei",
            "Abrothallus",
            "Abrothallus subhalei",
            "Fungi",
        ),
        (
            "midori",
            "KY290222.1.<1.>1539###root_1;Eukaryota_2759;Chordata_7711;Actinopteri_186623;Perciformes_8111;Serranidae_8214;Epinephelus_94232;Epinephelus coioides_94237",
            "Epinephelus",
            "Epinephelus coioides",
            None,
        ),
    ],
)
def test_reference_parsers(parser, header, genus, species, domain):
    lineage = references.PARSERS[parser](header)
    assert lineage.get("genus") == genus
    assert lineage.get("species") == species
    assert lineage.get("domain") == domain


def test_placeholders_are_not_names():
    assert references._clean("Nautilus") == "Nautilus"
    assert references._clean("NA") is None
    assert references._clean("Incertae_Sedis") is None
    assert references._clean("Dinophyceae_XX") is None


def test_stratified_split_keeps_every_class_in_training():
    labels = ["a"] * 10 + ["b"] * 3 + ["c"] * 1
    train, val, test = stratified_split(labels, (0.7, 0.15, 0.15))
    assert sorted(train + val + test) == list(range(len(labels)))
    assert {labels[i] for i in train} == {"a", "b", "c"}
    assert {labels[i] for i in val} >= {"a", "b"} and {labels[i] for i in test} >= {"a", "b"}


def test_alpha_diversity_known_values():
    d = alpha_diversity({"x": 10, "y": 10, "z": 10, "w": 10})
    assert d["observed"] == 4 and math.isclose(d["shannon"], math.log(4), rel_tol=1e-3)
    assert math.isclose(d["simpson"], 0.75) and math.isclose(d["pielou"], 1.0)
    # Two singletons, one doubleton: Chao1 = 5 + 2^2 / (2 * 1) = 7.
    assert alpha_diversity({"a": 1, "b": 1, "c": 2, "d": 5, "e": 9})["chao1"] == 7
    assert alpha_diversity({})["observed"] == 0


def test_threshold_calibration_and_f1():
    conf = np.array([0.99, 0.95, 0.9, 0.6, 0.5])
    correct = np.array([True, True, True, False, True])
    assert calibrate_threshold(conf, correct, 0.95) == 0.9
    assert calibrate_threshold(np.array([0.9]), np.array([False]), 0.95) == 0.8
    assert macro_f1(np.array([0, 1, 1]), np.array([0, 1, 0]), 2) == pytest.approx((2 / 3 + 2 / 3) / 2)

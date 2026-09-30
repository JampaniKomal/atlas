"""Synthetic reference databases and communities for the demo and the tests.

Sequences evolve down a small tree (root -> families -> genera -> species ->
individual sequences) by random substitutions and occasional indels, so
genera are distinct but related, like real marker genes. A sample mixes known
species at log-normal abundances with genera that are absent from the
reference: the "database gap" ATLAS exists for.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .fasta import Record

BASES = "ACGT"


def _random_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(BASES) for _ in range(n))


def mutate(seq: str, rate: float, rng: random.Random) -> str:
    out = []
    for b in seq:
        r = rng.random()
        if r < rate:
            out.append(rng.choice(BASES.replace(b, "") or BASES))
        elif r < rate * 1.05:
            continue  # deletion
        else:
            out.append(b)
            if rng.random() < rate * 0.05:
                out.append(rng.choice(BASES))  # insertion
    return "".join(out)


@dataclass
class Taxon:
    family: str
    genus: str
    species: list[tuple[str, str]]  # (name, sequence)


def make_tree(
    n_families: int,
    genera_per_family: int,
    species_per_genus: int,
    *,
    length: int,
    rng: random.Random,
    prefix: str = "",
) -> list[Taxon]:
    root = _random_seq(rng, length)
    taxa = []
    for f in range(n_families):
        family_seq = mutate(root, 0.22, rng)
        family = f"{prefix}Familia{chr(65 + f)}aceae"
        for g in range(genera_per_family):
            genus_seq = mutate(family_seq, 0.10, rng)
            genus = f"{prefix}Genus{chr(65 + f)}{g + 1}"
            species = [(f"{genus} sp{s + 1}", mutate(genus_seq, 0.03, rng)) for s in range(species_per_genus)]
            taxa.append(Taxon(family, genus, species))
    return taxa


def _header(i: int, t: Taxon, species: str, n: int) -> str:
    return f"SYN{i:05d}.1.{n} Bacteria;Synthetica;Synthetia;Synthetales;{t.family};{t.genus};{species}"


def make_reference(taxa: list[Taxon], per_species: int, rng: random.Random) -> list[Record]:
    records = []
    for t in taxa:
        for name, seq in t.species:
            for _ in range(per_species):
                s = mutate(seq, 0.01, rng)
                header = _header(len(records) + 1, t, name, len(s))
                records.append(Record(header.split()[0], header, s))
    return records


def make_sample(
    known: list[Taxon],
    novel: list[Taxon],
    reads: int,
    rng: random.Random,
    novel_fraction: float = 0.15,
    error_rate: float = 0.005,
) -> tuple[list[Record], dict[str, str]]:
    """Reads plus the true genus of each (prefixed "NOVEL:" for absent genera)."""
    pool = [(t, s) for t in known for s in t.species]
    novel_pool = [(t, s) for t in novel for s in t.species]
    weights = [rng.lognormvariate(0, 1.2) for _ in pool]
    records, truth = [], {}
    for i in range(reads):
        if novel_pool and rng.random() < novel_fraction:
            t, (_, seq) = rng.choice(novel_pool)
            label = f"NOVEL:{t.genus}"
        else:
            t, (_, seq) = rng.choices(pool, weights)[0]
            label = t.genus
        rid = f"read_{i + 1:05d}"
        records.append(Record(rid, rid, mutate(seq, error_rate, rng)))
        truth[rid] = label
    return records, truth

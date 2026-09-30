"""Per-marker defaults: which reference format to read, which lineages to keep,
and the k-mer size. Every value can be overridden on the command line."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Marker:
    name: str
    description: str
    parser: str
    k: int = 6
    domains: tuple[str, ...] = ()  # keep only these top-level taxa; empty keeps all
    min_length: int = 200


MARKERS: dict[str, Marker] = {
    m.name: m
    for m in (
        Marker("16S", "16S rRNA: bacteria and archaea (SILVA)", "silva", domains=("Bacteria", "Archaea")),
        Marker("18S", "18S rRNA: eukaryotes (PR2, or SILVA Eukaryota)", "pr2", domains=("Eukaryota",)),
        Marker("COI", "Cytochrome c oxidase I: animals (MIDORI2 or BOLD)", "midori", min_length=300),
        Marker("ITS", "Internal transcribed spacer: fungi (UNITE)", "unite", min_length=100),
    )
}


def get(name: str) -> Marker:
    key = name.upper()
    if key not in MARKERS:
        raise KeyError(f"unknown marker {name!r} (known: {', '.join(MARKERS)})")
    return MARKERS[key]

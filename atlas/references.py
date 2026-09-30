"""Parsers for the taxonomy in reference database FASTA headers.

Each parser turns a header into a lineage: a dict from rank name to taxon
name, with placeholder names ("uncultured", "Incertae_sedis", ...) removed so
they never become a class a model learns.

Supported formats:
  silva   SILVA SSU/LSU:   >AB001234.1.1500 Bacteria;Proteobacteria;...;Escherichia-Shigella;Escherichia coli
  pr2     PR2 v4 and v5:   >AB000912.1.1709_U|Eukaryota|TSAR|Alveolata|...|Heterocapsa|Heterocapsa_triquetra
  unite   UNITE general:   >Name|MT153946|SH1227328.10FU|refs|k__Fungi;p__Ascomycota;...;g__Abrothallus;s__Abrothallus_subhalei
  midori  MIDORI2 (COI):   >KY290222.1.<1.>1539###root_1;Eukaryota_2759;Chordata_7711;...;Genus_1234;Genus species_5678
  lineage generic:         >ID Rank1;Rank2;...;Genus;Species   (ranks counted from the right)
"""

from __future__ import annotations

import re
from collections.abc import Callable

Lineage = dict[str, str]

RANKS = ("domain", "phylum", "class", "order", "family", "genus", "species")

_PLACEHOLDER = re.compile(
    r"(?i)(?:uncultured.*|unidentified.*|unclassified.*|unknown.*|metagenome.*|environmental.*|"
    r"incertae[_ ]sedis.*|candidatus|.*_X+|.*[_ ]sp\.?|n/?a|none)"
)


def _clean(name: str) -> str | None:
    raw = name.strip().strip("_")
    if not raw or _PLACEHOLDER.fullmatch(raw) or _PLACEHOLDER.fullmatch(raw.replace(" ", "_")):
        return None
    return raw.replace("_", " ")


def _assign(pairs: list[tuple[str, str]]) -> Lineage:
    out: Lineage = {}
    for rank, raw in pairs:
        name = _clean(raw)
        if name:
            out[rank] = name
    return out


def parse_silva(header: str) -> Lineage:
    parts = header.split(maxsplit=1)
    if len(parts) < 2:
        return {}
    fields = [f for f in parts[1].split(";") if f.strip()]
    return _assign(list(zip(RANKS, fields, strict=False)))


_PR2_V5 = ("domain", "supergroup", "division", "subdivision", "class", "order", "family", "genus", "species")
_PR2_V4 = ("domain", "supergroup", "division", "class", "order", "family", "genus", "species")


_DOMAINS = {"Eukaryota", "Bacteria", "Archaea"}


def parse_pr2(header: str) -> Lineage:
    # The "taxo_long" files put gene, organelle and strain between the accession
    # and the taxonomy (>ACC|18S_rRNA|nucleus|strain|Eukaryota|...), so the
    # taxonomy is found by its first field, the domain.
    fields = header.split("|")[1:]
    start = next((i for i, f in enumerate(fields) if f in _DOMAINS), 0)
    fields = fields[start:]
    ranks = _PR2_V5 if len(fields) >= 9 else _PR2_V4
    return _assign(list(zip(ranks, fields, strict=False)))


_UNITE_PREFIX = {"k": "domain", "p": "phylum", "c": "class", "o": "order", "f": "family", "g": "genus", "s": "species"}


def parse_unite(header: str) -> Lineage:
    lineage = header.rsplit("|", 1)[-1]
    pairs = []
    for field in lineage.split(";"):
        prefix, _, name = field.strip().partition("__")
        if prefix in _UNITE_PREFIX:
            pairs.append((_UNITE_PREFIX[prefix], name))
    return _assign(pairs)


_TAXID = re.compile(r"_\d+$")


def parse_midori(header: str) -> Lineage:
    lineage = header.split("###", 1)[-1] if "###" in header else header.split(maxsplit=1)[-1]
    fields = [_TAXID.sub("", f) for f in lineage.split(";") if f.strip()]
    fields = [f for f in fields if f.lower() != "root"]
    return _from_right(fields)


def parse_lineage(header: str) -> Lineage:
    parts = header.split(maxsplit=1)
    fields = [f for f in (parts[1] if len(parts) > 1 else "").split(";") if f.strip()]
    return _from_right(fields)


def _from_right(fields: list[str]) -> Lineage:
    # The last field is the species and the one before it the genus; higher
    # ranks vary between databases, so only these are assigned by position.
    ranks = ("species", "genus", "family", "order", "class", "phylum")
    return _assign(list(zip(ranks, reversed(fields), strict=False)))


PARSERS: dict[str, Callable[[str], Lineage]] = {
    "silva": parse_silva,
    "pr2": parse_pr2,
    "unite": parse_unite,
    "midori": parse_midori,
    "lineage": parse_lineage,
}

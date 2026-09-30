"""Minimal streaming FASTA reader (plain or gzip) and writer."""

from __future__ import annotations

import gzip
import io
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    id: str
    description: str  # the full header line without ">"
    seq: str


def _parse(lines: Iterable[str]) -> Iterator[Record]:
    header: str | None = None
    chunks: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if header is not None:
                yield _record(header, chunks)
            header, chunks = line[1:].strip(), []
        elif header is not None:
            chunks.append(line)
    if header is not None:
        yield _record(header, chunks)


def _record(header: str, chunks: list[str]) -> Record:
    rid = header.split(maxsplit=1)[0] if header else ""
    return Record(id=rid, description=header, seq="".join(chunks).upper())


def read_fasta(path: str | Path) -> Iterator[Record]:
    """Yield the records of a FASTA file; `.gz` files are decompressed on the fly."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        yield from _parse(handle)


def parse_fasta(text: str) -> list[Record]:
    """Parse FASTA text. A bare sequence without a header becomes one record."""
    if not text.lstrip().startswith(">"):
        seq = "".join(text.split())
        return [Record("input_1", "input_1", seq.upper())] if seq else []
    return list(_parse(io.StringIO(text)))


def write_fasta(records: Iterable[Record], path: str | Path, width: int = 80) -> None:
    with open(path, "w", encoding="utf-8") as out:
        for r in records:
            out.write(f">{r.description or r.id}\n")
            for i in range(0, len(r.seq), width):
                out.write(r.seq[i : i + width] + "\n")

"""Reports: the framed text report of ATLAS v1, JSON, and a standalone HTML page."""

from __future__ import annotations

import html
import json
from dataclasses import asdict

from . import __version__
from .analysis import Result

WIDTH = 63


def _line(text: str = "") -> str:
    return f"| {text}"


def _bar() -> str:
    return "+" + "-" * (WIDTH - 2) + "+"


def _lineage(lineage: dict[str, str], keep=("phylum", "class", "order", "family")) -> str:
    return " > ".join(lineage[r] for r in keep if r in lineage)


def to_text(r: Result, top: int = 25) -> str:
    d = r.diversity
    out = [
        _bar(),
        "|          ATLAS: AI Taxonomic Learning & Analysis System     |",
        "|                         FINAL REPORT                        |",
        _bar(),
        "[  ENVIRONMENT & INPUT  ]",
        _bar(),
        _line(f"Compute: {r.device}"),
        _line(f"Input: {r.sample}"),
        _line(f"Sequences: {r.sequences} ({', '.join(f'{m}: {n}' for m, n in r.markers.items()) or 'none routed'})"),
        _line(f"Skipped (too short / no valid k-mers): {len(r.skipped)}"),
        _line(f"Analysis time: {r.seconds:.2f} s"),
        _bar(),
        "[  PART 1: FILTER RESULTS  ]",
        _bar(),
        _line(f"Classified: {r.classified} reads into {len(r.abundance)} taxa"),
    ]
    total = max(r.classified, 1)
    for taxon, n in list(r.abundance.items())[:top]:
        out.append(_line(f"  {taxon:<30} {n:>6}  {n / total:6.1%}   {_lineage(r.lineages.get(taxon, {}))}"))
    if len(r.abundance) > top:
        out.append(_line(f"  ... and {len(r.abundance) - top} more taxa"))
    out += [
        _bar(),
        "[  PART 2: EXPLORER RESULTS  ]",
        _bar(),
        _line(f"Unclassified: {r.unclassified} reads -> {len(r.clusters)} clusters, {len(r.noise)} unclustered"),
    ]
    for c in r.clusters:
        flag = "PUTATIVE NOVEL LINEAGE" if c.novel else "close to a known taxon"
        out.append(_line(f"  {c.id:<8} {c.size:>5} reads  nearest {c.nearest} (similarity {c.similarity:.3f})  {flag}"))
        out.append(_line(f"           representative: {c.representative}"))
    out += [
        _bar(),
        "[  PART 3: BIODIVERSITY  ]",
        _bar(),
        _line(f"Observed units (taxa + clusters): {d['observed']}    Chao1 estimate: {d['chao1']}"),
        _line(f"Shannon H': {d['shannon']}    Gini-Simpson: {d['simpson']}    Pielou evenness: {d['pielou']}"),
        _bar(),
    ]
    return "\n".join(out)


def to_dict(r: Result) -> dict:
    return {
        "atlas_version": __version__,
        "sample": r.sample,
        "sequences": r.sequences,
        "classified": r.classified,
        "unclassified": r.unclassified,
        "skipped": len(r.skipped),
        "markers": r.markers,
        "models": r.models,
        "abundance": [{"taxon": t, "reads": n, "lineage": r.lineages.get(t, {})} for t, n in r.abundance.items()],
        "clusters": [c.to_dict() for c in r.clusters],
        "unclustered": r.noise,
        "diversity": r.diversity,
        "assignments": [asdict(a) for a in r.assignments],
        "seconds": r.seconds,
        "device": r.device,
    }


def to_json(r: Result) -> str:
    return json.dumps(to_dict(r), indent=2)


def _bars(items: list[tuple[str, int, bool]]) -> str:
    if not items:
        return "<p class='muted'>Nothing to plot.</p>"
    top = max(n for _, n, _ in items)
    row_h, label_w, width = 22, 230, 720
    rows = []
    for i, (label, n, novel) in enumerate(items):
        w = (width - label_w - 60) * n / top
        y = i * row_h
        cls = "bar novel" if novel else "bar"
        rows.append(
            f"<text x='{label_w - 8}' y='{y + 15}' text-anchor='end'>{html.escape(label[:34])}</text>"
            f"<rect class='{cls}' x='{label_w}' y='{y + 3}' width='{w:.1f}' height='{row_h - 6}' rx='3'/>"
            f"<text x='{label_w + w + 6:.1f}' y='{y + 15}'>{n}</text>"
        )
    return f"<svg viewBox='0 0 {width} {len(items) * row_h}' role='img' aria-label='Reads per taxon'>" + "".join(rows) + "</svg>"


def to_html(r: Result, top: int = 30) -> str:
    esc = html.escape
    d = r.diversity
    items = [(t, n, False) for t, n in list(r.abundance.items())[:top]]
    items += [(f"{c.id} near {c.nearest}", c.size, c.novel) for c in r.clusters]
    items.sort(key=lambda x: -x[1])
    taxa_rows = "".join(
        f"<tr><td><i>{esc(t)}</i></td><td class='n'>{n}</td><td class='n'>{n / max(r.classified, 1):.1%}</td>"
        f"<td>{esc(_lineage(r.lineages.get(t, {})))}</td></tr>"
        for t, n in r.abundance.items()
    )
    cluster_rows = "".join(
        f"<tr><td>{esc(c.id)}</td><td class='n'>{c.size}</td><td><i>{esc(c.nearest)}</i><br>"
        f"<span class='muted'>{esc(_lineage(c.nearest_lineage))}</span></td><td class='n'>{c.similarity:.3f}</td>"
        f"<td>{'<b class=novel-tag>putative novel</b>' if c.novel else 'near known taxon'}</td>"
        f"<td><details><summary>{esc(c.representative)}</summary><code>{esc(c.representative_seq)}</code></details></td></tr>"
        for c in r.clusters
    )
    cards = [
        ("Sequences", r.sequences),
        ("Classified", r.classified),
        ("Explorer clusters", len(r.clusters)),
        ("Observed units", d["observed"]),
        ("Chao1", d["chao1"]),
        ("Shannon H'", d["shannon"]),
        ("Gini-Simpson", d["simpson"]),
        ("Pielou", d["pielou"]),
    ]
    card_html = "".join(f"<div class='card'><div class='v'>{v}</div><div class='l'>{k}</div></div>" for k, v in cards)
    models = ", ".join(f"{m} ({v['classes']} {v['rank']} classes, k={v['k']})" for m, v in r.models.items())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ATLAS report: {esc(r.sample)}</title>
<style>
:root {{ --bg:#fbfbf9; --fg:#1d232a; --muted:#66707a; --line:#dde2e6; --card:#fff; --bar:#1f7a8c; --novel:#c2572b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#11161b; --fg:#e4e8ec; --muted:#98a3ad; --line:#26313a; --card:#161d23; --bar:#4fb3c7; --novel:#f08a5d; }} }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1000px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin: 36px 0 12px; }}
.muted {{ color: var(--muted); }}
.cards {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-top: 20px; }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }}
.card .v {{ font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }} .card .l {{ color: var(--muted); font-size: 13px; }}
svg {{ width: 100%; height: auto; font-size: 12px; fill: var(--fg); }} .bar {{ fill: var(--bar); }} .bar.novel {{ fill: var(--novel); }}
.scroll {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; }} th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font-size: 13px; color: var(--muted); font-weight: 600; }} td.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
code {{ display:block; max-width: 420px; word-break: break-all; font-size: 12px; }}
.novel-tag {{ color: var(--novel); }}
</style></head>
<body><main>
<h1>ATLAS report: {esc(r.sample)}</h1>
<div class="muted">ATLAS {__version__} · {esc(models)} · {r.seconds:.2f} s on {esc(r.device)}</div>
<div class="cards">{card_html}</div>
<h2>Community composition</h2>
<p class="muted">Classified taxa and Explorer clusters by read count. <span class="novel-tag">Orange</span> clusters are farther from every known taxon than 95% of reference sequences are from their own.</p>
{_bars(items)}
<h2>Classified taxa (Filter)</h2>
<div class="scroll"><table><thead><tr><th>Taxon</th><th>Reads</th><th>Share</th><th>Lineage</th></tr></thead><tbody>{taxa_rows or "<tr><td colspan=4 class=muted>None</td></tr>"}</tbody></table></div>
<h2>Unclassified reads (Explorer)</h2>
<p class="muted">{r.unclassified} reads below the confidence threshold: {len(r.clusters)} clusters, {len(r.noise)} left unclustered. Representative sequences are the members closest to each cluster's centroid; confirm them with BLAST or phylogenetic placement before naming anything.</p>
<div class="scroll"><table><thead><tr><th>Cluster</th><th>Reads</th><th>Nearest known taxon</th><th>Similarity</th><th>Assessment</th><th>Representative</th></tr></thead><tbody>{cluster_rows or "<tr><td colspan=6 class=muted>None</td></tr>"}</tbody></table></div>
</main></body></html>
"""

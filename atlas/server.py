"""`atlas serve`: the ATLAS web interface.

Serves the upload page from ATLAS v1 (web/index.html) and the
`POST /run_analysis` endpoint it was written against, which v1 removed along
with its HTTP server. Binds to 127.0.0.1 by default.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from importlib import resources
from pathlib import Path

from .analysis import analyze, load_models
from .fasta import parse_fasta
from .report import to_dict, to_html, to_text

MAX_UPLOAD = 64 * 1024 * 1024


def create_app(models_dir: str | Path, min_cluster_size: int = 5):
    from flask import Flask, Response, jsonify, request

    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
    state: dict = {"models": None}
    reports: OrderedDict[str, str] = OrderedDict()

    def models():
        if state["models"] is None:
            state["models"] = load_models(models_dir)
        return state["models"]

    @app.get("/")
    def index():
        page = resources.files("atlas.web").joinpath("index.html").read_text(encoding="utf-8")
        return Response(page, mimetype="text/html")

    @app.get("/health")
    def health():
        return jsonify(status="ok", models=[m.name for m in models()])

    @app.post("/run_analysis")
    def run_analysis():
        upload = request.files.get("file")
        if upload and upload.filename:
            text, sample = upload.read().decode("utf-8", errors="replace"), Path(upload.filename).name
        else:
            text, sample = request.form.get("data", ""), "pasted sequence"
        records = parse_fasta(text)
        if not records:
            return jsonify(status="error", message="No sequences found. Upload FASTA or paste a DNA sequence."), 400
        try:
            result = analyze(records, models(), sample=sample, min_cluster_size=min_cluster_size)
        except ValueError as e:
            return jsonify(status="error", message=str(e)), 400
        key = uuid.uuid4().hex
        reports[key] = to_html(result)
        while len(reports) > 20:
            reports.popitem(last=False)
        summary = to_dict(result)
        summary.pop("assignments")
        return jsonify(
            status="success",
            report_content=to_text(result),
            classified_results=result.chart_data(),
            report=summary,
            html_report=f"/reports/{key}",
        )

    @app.get("/reports/<key>")
    def report(key: str):
        if key not in reports:
            return Response("Report expired; run the analysis again.", status=404, mimetype="text/plain")
        return Response(reports[key], mimetype="text/html")

    return app

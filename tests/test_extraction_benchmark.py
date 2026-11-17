"""Tests for the extraction benchmark: rendering is deterministic and the report merge keeps other keys."""

from __future__ import annotations

import json

from src.ingestion import extraction_benchmark as eb


def test_render_texts_deterministic() -> None:
    assert eb.render_texts(5, seed=1) == eb.render_texts(5, seed=1)
    assert len(eb.render_texts(5)) == 5


def test_merge_report_keeps_other_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(eb, "write_report", lambda name, release, results, seed=None: _write(tmp_path, results))
    (tmp_path / "v1_2_extraction.json").write_text(json.dumps({"results": {"other": 1, "throughput": {"x": 0}}}))
    eb.merge_report({"y": 2}, reports_dir=tmp_path)
    saved = json.loads((tmp_path / "v1_2_extraction.json").read_text())["results"]
    assert saved == {"other": 1, "throughput": {"y": 2}}


def _write(tmp_path, results):
    path = tmp_path / "v1_2_extraction.json"
    path.write_text(json.dumps({"results": results}))
    return path

"""Tests for the release gate runner and report writer."""

from __future__ import annotations

import json

import pytest

from src.release import gates, reporting
from src.release.gates import GATES, GATE_ID_PATTERN, GateResult, run_gates


def test_unknown_version_raises():
    with pytest.raises(ValueError):
        run_gates("9.9")


def test_gate_ids_unique_and_match_regex():
    ids = []
    for fns in GATES.values():
        for fn in fns:
            res = fn()
            assert GATE_ID_PATTERN.match(res.gate_id)
            ids.append(res.gate_id)
    assert len(ids) == len(set(ids))


def test_raising_gate_reported_as_failed(monkeypatch):
    def boom() -> GateResult:
        raise RuntimeError("kaboom")

    monkeypatch.setitem(GATES, "1.1", [boom])
    results = run_gates("1.1")
    assert len(results) == 1 and not results[0].passed
    assert "kaboom" in results[0].detail


def test_all_runs_every_version(monkeypatch):
    ok = lambda: GateResult("G1.1-01", "H", True, "ok")
    monkeypatch.setattr(gates, "GATES", {"1.1": [ok], "1.2": [ok]})
    assert len(run_gates("all")) == 2


def test_cli_exit_codes(monkeypatch, capsys):
    monkeypatch.setitem(GATES, "1.1", [lambda: GateResult("G1.1-01", "H", True, "fine")])
    assert gates.main(["--version", "1.1"]) == 0
    assert "PASS G1.1-01 [H] fine" in capsys.readouterr().out
    monkeypatch.setitem(GATES, "1.1", [lambda: GateResult("G1.1-01", "H", False, "bad")])
    assert gates.main(["--version", "1.1"]) == 1
    assert gates.main(["--version", "zzz"]) == 1


def test_write_report_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "REPORTS_DIR", tmp_path / "reports")
    p = reporting.write_report("topic", "1.1", {"a": 1})
    data = json.loads(p.read_text())
    assert set(data) == {"schema_version", "release", "generated_at", "git_commit",
                         "seed", "results", "tripwire_notes"}
    assert data["schema_version"] == 1 and data["seed"] == 42
    assert data["results"] == {"a": 1} and data["tripwire_notes"] == []
    assert reporting.write_report("t2", "1.1", {}, seed=7).exists()
    with pytest.raises(ValueError):
        reporting.write_report("../x", "1.1", {})

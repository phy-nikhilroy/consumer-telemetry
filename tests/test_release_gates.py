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


def test_v1_3_gates_are_registered_with_their_types():
    """G1.3-01 to G1.3-12 are registered once each; 05, 06 and 11 are tripwires."""
    ids = [fn.gate_id for fn in gates.GATES["1.3"]]
    assert ids == [f"G1.3-{i:02d}" for i in range(1, 13)]
    kinds = {fn.gate_id: fn.kind for fn in gates.GATES["1.3"]}
    assert {g for g, k in kinds.items() if k == "T"} == {"G1.3-05", "G1.3-06", "G1.3-11"}


def test_suite_seconds_reads_the_pytest_summary():
    assert gates._suite_seconds("779 passed, 15 warnings in 224.38s (0:03:44)") == 224.38
    assert gates._suite_seconds("no output") == float("inf")


def test_feature_sql_gate_passes_on_this_checkout():
    assert gates.feature_sql_has_no_avg_price().passed


def test_v1_4_gates_are_registered_with_their_types():
    """G1.4-01 to G1.4-15 are registered once each; 05 to 09, 12 and 13 are tripwires."""
    ids = [fn.gate_id for fn in GATES["1.4"]]
    assert ids == [f"G1.4-{i:02d}" for i in range(1, 16)]
    kinds = {fn.gate_id: fn.kind for fn in GATES["1.4"]}
    tripwires = {f"G1.4-{i:02d}" for i in (5, 6, 7, 8, 9, 12, 13)}
    assert {g for g, k in kinds.items() if k == "T"} == tripwires


def test_composite_gates_skip_each_other(monkeypatch):
    """G1.3-12 and G1.4-15 both run every other gate, so each must skip both or they recurse."""
    assert gates._COMPOSITE_GATES == {"G1.3-12", "G1.4-15"}
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(gates, "_suite_result", lambda: (900, 0, "900 passed in 200.00s"))
    calls = []

    def stub(gid):
        def fake():
            calls.append(gid)
            return GateResult(gid, "H", True, "ok")
        fake.gate_id = gid
        return fake

    for version in list(GATES):
        replaced = [fn if fn.gate_id in gates._COMPOSITE_GATES else stub(fn.gate_id) for fn in GATES[version]]
        monkeypatch.setitem(GATES, version, replaced)
    for gid in ("G1.3-12", "G1.4-15"):
        composite = next(fn for v in GATES for fn in GATES[v] if fn.gate_id == gid)
        assert composite().passed, gid
    assert not set(calls) & gates._COMPOSITE_GATES


def test_simulation_grep_gate_passes_on_this_checkout():
    assert gates.simulation_has_no_module_random_or_archetype_loop().passed

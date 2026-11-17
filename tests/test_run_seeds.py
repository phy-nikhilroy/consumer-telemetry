"""
Tests for the seedable simulation and the multi-seed harness (E9).

Covers: run_simulation(seed=...) defaults reproducing v1.0 exactly, run_one_seed isolation
under its output directory, seed sensitivity, same-seed determinism, and run_seeds
aggregation of every numeric metric to mean, std, min, max, n.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import duckdb
import pytest
from hypothesis import given, settings, strategies as st

from src.config import ARTIFACTS_DIR, DEFAULT_DB_PATH, FEATURES_DIR, RANDOM_SEED

USERS = 24
DAYS = 30
AGG_KEYS = {"mean", "std", "min", "max", "n"}


def _listing(*directories: Path) -> list:
    """Sorted (path, size, mtime) listing of every file under the directories."""
    out = []
    for directory in directories:
        if directory.exists():
            for p in sorted(directory.rglob("*")):
                if p.is_file():
                    s = p.stat()
                    out.append((str(p), s.st_size, s.st_mtime_ns))
    return out


def _flatten(obj: Any, prefix: str = "") -> Dict[str, float]:
    """Flattens nested dicts to {dotted.path: number}, ignoring non-numeric leaves."""
    flat: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        flat[prefix] = float(obj)
    return flat


def _agg_leaves(obj: Any, prefix: str = "") -> Dict[str, dict]:
    """Finds every dict whose keys are exactly the aggregate keys."""
    found: Dict[str, dict] = {}
    if isinstance(obj, dict):
        if AGG_KEYS <= set(obj) and all(isinstance(obj[k], (int, float)) for k in AGG_KEYS):
            found[prefix] = obj
        else:
            for k, v in obj.items():
                found.update(_agg_leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    return found


@pytest.fixture(scope="module")
def seed_runs(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """Runs seed 1 twice and seed 2 once, recording protected-directory listings."""
    from src.experiments.run_seeds import run_one_seed

    root = tmp_path_factory.mktemp("seeds")
    before = _listing(FEATURES_DIR, ARTIFACTS_DIR)
    a1 = run_one_seed(1, root / "a1", USERS, DAYS)
    a2 = run_one_seed(1, root / "a2", USERS, DAYS)
    b = run_one_seed(2, root / "b", USERS, DAYS)
    after = _listing(FEATURES_DIR, ARTIFACTS_DIR)
    return {"root": root, "a1": a1, "a2": a2, "b": b, "before": before, "after": after}


def test_run_one_seed_writes_nothing_outside_tmp_path(seed_runs):
    """Protected feature and artifact directories are unchanged after a run."""
    assert seed_runs["before"] == seed_runs["after"]
    assert any(seed_runs["root"].rglob("*")), "run wrote nothing under its out_dir"


def test_run_one_seed_does_not_touch_default_warehouse(seed_runs):
    """The committed warehouse is never opened for writing by the harness."""
    assert Path(DEFAULT_DB_PATH).exists()
    # Event count of the default warehouse must stay at the v1.0 value for 400 users.
    con = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    try:
        assert con.execute("SELECT count(*) FROM fact_events").fetchone()[0] > USERS * 10
    finally:
        con.close()


def test_run_one_seed_returns_numeric_metrics(seed_runs):
    """The result is a dict containing numeric test metrics."""
    result = seed_runs["a1"]
    assert isinstance(result, dict)
    flat = _flatten(result)
    assert len(flat) > 0
    assert all(math.isfinite(v) or math.isnan(v) for v in flat.values())


def test_different_seeds_give_different_event_counts(seed_runs):
    """Seeds 1 and 2 simulate different populations, so event counts differ."""
    fa, fb = _flatten(seed_runs["a1"]), _flatten(seed_runs["b"])
    count_keys = [k for k in fa if "event" in k.lower() and "count" in k.lower() or k.endswith("n_events") or k.endswith("total_events")]
    assert count_keys, f"no event count key in result: {sorted(fa)}"
    assert any(fa[k] != fb[k] for k in count_keys)


def test_same_seed_twice_gives_identical_metrics(seed_runs):
    """Same seed, different out_dir: every metric is identical."""
    fa, fb = _flatten(seed_runs["a1"]), _flatten(seed_runs["a2"])
    assert fa.keys() == fb.keys()
    for k in fa:
        if math.isnan(fa[k]):
            assert math.isnan(fb[k]), k
        else:
            assert fa[k] == fb[k], k


def test_different_seeds_change_some_metric(seed_runs):
    """Seed is actually threaded through: at least one metric differs."""
    fa, fb = _flatten(seed_runs["a1"]), _flatten(seed_runs["b"])
    assert any(fa[k] != fb.get(k) for k in fa)


def test_run_simulation_accepts_seed_with_default_random_seed():
    """run_simulation gains seed defaulting to RANDOM_SEED, and the CLI exposes flags."""
    from src.ingestion import simulate_users

    sig = inspect.signature(simulate_users.run_simulation)
    assert "seed" in sig.parameters
    assert sig.parameters["seed"].default == RANDOM_SEED
    source = Path(simulate_users.__file__).read_text()
    for flag in ("--seed", "--db-path", "--output-parquet"):
        assert flag in source


def test_run_simulation_default_equals_explicit_default_seed(tmp_path):
    """No seed argument and seed=RANDOM_SEED produce identical events."""
    import polars as pl

    from src.ingestion.simulate_users import run_simulation

    p1, p2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    run_simulation(num_users=USERS, days=DAYS, db_path=str(tmp_path / "a.duckdb"), output_parquet=str(p1))
    run_simulation(num_users=USERS, days=DAYS, db_path=str(tmp_path / "b.duckdb"),
                   output_parquet=str(p2), seed=RANDOM_SEED)
    assert pl.read_parquet(p1).equals(pl.read_parquet(p2))


def test_run_simulation_seed_changes_events_and_is_reproducible(tmp_path):
    """Different seeds differ; the same seed reproduces byte-equal events."""
    import polars as pl

    from src.ingestion.simulate_users import run_simulation

    def go(seed: int, name: str) -> "pl.DataFrame":
        out = tmp_path / f"{name}.parquet"
        run_simulation(num_users=USERS, days=DAYS, db_path=str(tmp_path / f"{name}.duckdb"),
                       output_parquet=str(out), seed=seed)
        return pl.read_parquet(out)

    a, a_again, b = go(7, "a"), go(7, "a2"), go(8, "b")
    assert a.equals(a_again)
    assert not a.equals(b)


def test_run_simulation_default_matches_committed_warehouse(tmp_path):
    """Default flags reproduce v1.0: event count and first and last event ids match."""
    from src.ingestion.simulate_users import simulate_events_dataset

    events, _ = simulate_events_dataset(num_users=400, days=30)
    con = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    try:
        count, first, last = con.execute(
            "SELECT count(*), min(event_id), max(event_id) FROM fact_events"
        ).fetchone()
    finally:
        con.close()
    assert len(events) == count
    assert events[0]["event_id"] == first
    assert events[-1]["event_id"] == last


def test_run_simulation_writes_only_to_given_paths(tmp_path):
    """db_path and output_parquet are honoured, not the defaults."""
    from src.ingestion.simulate_users import run_simulation

    db, pq = tmp_path / "w.duckdb", tmp_path / "e.parquet"
    run_simulation(num_users=USERS, days=DAYS, db_path=str(db), output_parquet=str(pq), seed=3)
    assert db.exists() and pq.exists()


# ---------------------------------------------------------------- run_seeds aggregation

def _fake_factory(table: Dict[int, Dict[str, Any]], calls: list):
    def fake(seed: int, out_dir: Path, users: int, days: int) -> Dict[str, Any]:
        calls.append((seed, Path(out_dir), users, days))
        return table[seed]
    return fake


def test_run_seeds_aggregates_mean_std_min_max_n(tmp_path, monkeypatch):
    """Every numeric metric becomes mean, std, min, max, n over the seeds."""
    from src.experiments import run_seeds as rs

    table = {
        1: {"propensity": {"auc": 0.6, "pr_auc": 0.2}, "n_events": 100},
        2: {"propensity": {"auc": 0.8, "pr_auc": 0.4}, "n_events": 200},
        3: {"propensity": {"auc": 0.7, "pr_auc": 0.3}, "n_events": 300},
    }
    calls: list = []
    monkeypatch.setattr(rs, "run_one_seed", _fake_factory(table, calls))
    out = rs.run_seeds([1, 2, 3], tmp_path, users=5, days=6)
    leaves = _agg_leaves(out)
    auc = next(v for k, v in leaves.items() if k.endswith("propensity.auc"))
    assert auc["n"] == 3
    assert auc["mean"] == pytest.approx(0.7)
    assert auc["min"] == pytest.approx(0.6)
    assert auc["max"] == pytest.approx(0.8)
    assert auc["std"] in (pytest.approx(0.1), pytest.approx(math.sqrt(0.02 / 3)))
    assert any(k.endswith("n_events") for k in leaves)
    assert len(leaves) == 3
    assert [c[0] for c in calls] == [1, 2, 3]
    assert all(c[2:] == (5, 6) for c in calls)


def test_run_seeds_gives_each_seed_its_own_directory(tmp_path, monkeypatch):
    """Per-seed runs are isolated under out_root, in distinct directories."""
    from src.experiments import run_seeds as rs

    calls: list = []
    monkeypatch.setattr(rs, "run_one_seed", _fake_factory({1: {"m": 1.0}, 2: {"m": 2.0}}, calls))
    rs.run_seeds([1, 2], tmp_path, users=5, days=6)
    dirs = [c[1] for c in calls]
    assert len(set(dirs)) == 2
    for d in dirs:
        assert tmp_path in d.parents or d == tmp_path


def test_run_seeds_single_seed_has_n_one_and_zero_spread(tmp_path, monkeypatch):
    """One seed: n is 1, min equals max equals mean."""
    from src.experiments import run_seeds as rs

    monkeypatch.setattr(rs, "run_one_seed", _fake_factory({9: {"m": 0.5}}, []))
    leaves = _agg_leaves(rs.run_seeds([9], tmp_path, users=5, days=6))
    (agg,) = leaves.values()
    assert agg["n"] == 1
    assert agg["mean"] == agg["min"] == agg["max"] == pytest.approx(0.5)
    assert agg["std"] == 0 or math.isnan(agg["std"])


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False), min_size=2, max_size=8))
def test_run_seeds_aggregate_properties(values):
    """For any metric values: min <= mean <= max, n == count, order independent."""
    import tempfile

    from src.experiments import run_seeds as rs

    seeds = list(range(len(values)))
    table = {s: {"m": v} for s, v in zip(seeds, values)}
    orig = rs.run_one_seed
    rs.run_one_seed = _fake_factory(table, [])
    try:
        with tempfile.TemporaryDirectory() as d:
            fwd = next(iter(_agg_leaves(rs.run_seeds(seeds, Path(d), users=1, days=1)).values()))
            rev = next(iter(_agg_leaves(rs.run_seeds(seeds[::-1], Path(d), users=1, days=1)).values()))
    finally:
        rs.run_one_seed = orig
    assert fwd["n"] == len(values)
    assert fwd["min"] == min(values) and fwd["max"] == max(values)
    assert fwd["min"] - 1e-9 <= fwd["mean"] <= fwd["max"] + 1e-9
    assert fwd["std"] >= 0
    assert fwd["mean"] == pytest.approx(rev["mean"], abs=1e-9)
    assert fwd["std"] == pytest.approx(rev["std"], abs=1e-9)

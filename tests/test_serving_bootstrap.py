"""Tests for the serving bootstrap sizes, tiny fixture sizes, and the saved profiles file."""
from __future__ import annotations

import dataclasses
import inspect
import json
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl
from hypothesis import given, settings, strategies as st

from src import config
from src.api import warehouse as serving_warehouse
from src.ingestion import simulate_users
from src.simulation.behavior import UserProfile, UserState, profiles_frame, read_profiles
from tests import conftest

START = datetime(2026, 9, 28, 8, 0)
RAW = ["event_id", "user_id", "timestamp", "app_name", "raw_text"]


def _engine_run(seed: int, users: int, days: int):
    engine = simulate_users.build_engine(days=days + 4, seed=seed)
    profiles = engine.create_profiles(users)
    events = engine.simulate_events(profiles, START, days)
    return engine, profiles, events


def test_serving_simulation_constants():
    assert config.SERVING_SIMULATION_USERS == 300
    assert config.SERVING_SIMULATION_DAYS == 45


def test_tiny_fixture_constants():
    assert conftest.TINY_USERS == 40
    assert conftest.TINY_DAYS == 30


def test_serving_bootstrap_defaults_follow_config():
    params = inspect.signature(serving_warehouse.ensure_serving_warehouse).parameters
    assert params["num_users"].default == config.SERVING_SIMULATION_USERS
    assert params["days"].default == config.SERVING_SIMULATION_DAYS


def test_tiny_warehouse_population_matches_tiny_users(tiny_warehouse):
    con = duckdb.connect(str(tiny_warehouse), read_only=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0] == conftest.TINY_USERS
    finally:
        con.close()


def test_tiny_warehouse_has_no_profile_or_truth_columns(tiny_warehouse):
    con = duckdb.connect(str(tiny_warehouse), read_only=True)
    try:
        cols = [r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns").fetchall()]
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()]
    finally:
        con.close()
    assert "archetype" not in cols
    assert not any(c.startswith("true_") for c in cols)
    assert not any("profile" in t for t in tables)


def test_default_profiles_path_is_beside_truth_with_db_stem():
    p = simulate_users.default_profiles_path(Path("/x/db/consumer.duckdb"), Path("/y/truth/sim_truth.parquet"))
    assert p == Path("/y/truth/consumer_profiles.parquet")
    p2 = simulate_users.default_profiles_path("/x/other.duckdb", "/y/z.parquet")
    assert p2 == Path("/y/other_profiles.parquet")


def test_profiles_frame_one_row_per_user_with_all_fields():
    engine, profiles, _ = _engine_run(3, 8, 5)
    frame = profiles_frame(profiles, engine.states)
    assert frame.height == len(profiles)
    assert frame["user_id"].n_unique() == len(profiles)
    for f in dataclasses.fields(UserProfile) + dataclasses.fields(UserState):
        assert f.name in frame.columns, f.name
    row = frame.row(0, named=True)
    assert isinstance(row["category_weights"], str)
    assert isinstance(row["recent_views"], str)
    assert json.loads(row["category_weights"]) == profiles[0].category_weights
    assert json.loads(row["recent_views"]) == engine.states[profiles[0].user_id].recent_views


@settings(max_examples=6, deadline=None)
@given(seed=st.integers(0, 5000), users=st.integers(1, 6), days=st.integers(1, 6))
def test_property_profiles_round_trip(seed, users, days):
    engine, profiles, _ = _engine_run(seed, users, days)
    states = {p.user_id: engine.states[p.user_id] for p in profiles}
    back_profiles, back_states = read_profiles_via_file(profiles, states)
    assert back_profiles == list(profiles)
    assert back_states == states


def read_profiles_via_file(profiles, states, _dir=[None]):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.parquet"
        profiles_frame(profiles, states).write_parquet(path)
        return read_profiles(path)


def test_round_trip_preserves_dict_and_optional_fields():
    engine, profiles, _ = _engine_run(5, 10, 20)
    states = dict(engine.states)
    assert any(s.recent_views for s in states.values())
    back_p, back_s = read_profiles_via_file(profiles, states)
    assert back_p == list(profiles)
    for uid, s in states.items():
        assert back_s[uid] == s
        assert type(back_s[uid].origin) is datetime
        assert back_s[uid].churned_day == s.churned_day


def test_profiles_file_continues_users_like_longer_simulation(tmp_path):
    seed, users = 12, 12
    engine, profiles, _ = _engine_run(seed, users, 10)
    path = tmp_path / "consumer_profiles.parquet"
    profiles_frame(profiles, engine.states).write_parquet(path)
    loaded_profiles, loaded_states = read_profiles(path)

    long_engine = simulate_users.build_engine(days=14 + 4, seed=seed)
    long_profiles = long_engine.create_profiles(users)
    long_events = long_engine.simulate_events(long_profiles, START, 14)
    t0, t1 = START + timedelta(days=10), START + timedelta(days=14)
    checked = 0
    fresh = simulate_users.build_engine(days=14 + 4, seed=seed)
    for p in loaded_profiles:
        got = fresh.continue_user(p, loaded_states[p.user_id], t0, t1)
        got = sorted(got, key=lambda e: (e["timestamp"], e["event_id"]))
        want = sorted((e for e in long_events if e["user_id"] == p.user_id and e["timestamp"] >= t0),
                      key=lambda e: (e["timestamp"], e["event_id"]))
        assert len(got) == len(want), p.user_id
        for g, w in zip(got, want):
            for k in RAW:
                assert g[k] == w[k], (p.user_id, k)
        checked += len(want)
    assert checked > 0


def test_run_simulation_writes_profiles_beside_truth(tmp_path):
    out = tmp_path / "events.parquet"
    truth = tmp_path / "truth_dir" / "t.parquet"
    truth.parent.mkdir()
    db = tmp_path / "consumer.duckdb"
    result = simulate_users.run_simulation(num_users=6, days=5, db_path=str(db),
                                           output_parquet=str(out), truth_parquet=str(truth), workers=1)
    path = Path(result["profiles_path"])
    assert path == truth.parent / "consumer_profiles.parquet"
    assert path == simulate_users.default_profiles_path(db, truth)
    assert path.exists()
    profiles, states = read_profiles(path)
    assert len(profiles) == 6 and set(states) == {p.user_id for p in profiles}
    assert all(isinstance(p, UserProfile) for p in profiles)
    assert all(s.next_day == 5 for s in states.values())
    assert pl.read_parquet(path).height == 6

"""
Tests for the simulation command line front end over the v1.4 behaviour engine.

The engine's mechanics are tested in `tests/test_behavior.py`; these tests cover what this
module adds: config-driven defaults, event dicts in timestamp order, the raw and truth split,
the price beta pass-through, the warehouse reload, and the removal of the v1.0 engine.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import inspect
from pathlib import Path

import duckdb
import polars as pl
import pytest

from src.config import PRICE_BETA, SIMULATION_DAYS, SIMULATION_START, SIMULATION_USERS
from src.ingestion import simulate_users
from src.ingestion.simulate_users import (
    RAW_EVENT_COLUMNS,
    TRUTH_COLUMNS,
    reload_duckdb_warehouse,
    simulate_events_dataset,
    split_events,
)


@pytest.fixture(scope="module")
def small():
    """Twelve users over five days, seed 100."""
    return simulate_events_dataset(num_users=12, days=5, seed=100)


def test_defaults_come_from_config():
    for fn in (simulate_users.simulate_events_dataset, simulate_users.run_simulation):
        params = inspect.signature(fn).parameters
        assert params["num_users"].default == SIMULATION_USERS
        assert params["days"].default == SIMULATION_DAYS
        assert params["price_beta"].default == PRICE_BETA


def test_events_carry_raw_and_truth_keys_in_time_order(small):
    events, profiles = small
    assert len(profiles) == 12
    assert len(events) > 12
    for key in RAW_EVENT_COLUMNS + TRUTH_COLUMNS:
        assert key in events[0]
    keys = [(e["timestamp"], e["event_id"]) for e in events]
    assert keys == sorted(keys)
    assert len({e["event_id"] for e in events}) == len(events)


def test_default_start_is_simulation_start(small):
    events, _ = small
    start = datetime.fromisoformat(SIMULATION_START)
    assert min(e["timestamp"] for e in events) >= start
    assert max(e["timestamp"] for e in events) < start + timedelta(days=6)


def test_aware_start_date_is_converted_to_naive_utc():
    aware = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
    events, _ = simulate_events_dataset(num_users=4, days=2, seed=3, start_date=aware)
    assert all(e["timestamp"].tzinfo is None for e in events)
    assert min(e["timestamp"] for e in events) >= aware.replace(tzinfo=None)


def test_same_seed_same_events_and_seed_changes_them(small):
    events, _ = small
    again, _ = simulate_events_dataset(num_users=12, days=5, seed=100)
    other, _ = simulate_events_dataset(num_users=12, days=5, seed=101)
    assert events == again
    assert [e["raw_text"] for e in events] != [e["raw_text"] for e in other]


def test_split_events_separates_raw_from_truth(small):
    events, _ = small
    raw, truth = split_events(events)
    assert tuple(raw.columns) == tuple(RAW_EVENT_COLUMNS)
    assert tuple(truth.columns) == tuple(TRUTH_COLUMNS)
    assert raw["event_id"].to_list() == truth["event_id"].to_list()
    assert raw.schema["timestamp"] == pl.Datetime("us")


def test_price_beta_reaches_the_engine(monkeypatch):
    seen = {}
    real = simulate_users.BehaviorEngine.__init__

    def spy(self, *args, **kwargs):
        seen["price_beta"] = kwargs.get("price_beta")
        real(self, *args, **kwargs)

    monkeypatch.setattr(simulate_users.BehaviorEngine, "__init__", spy)
    simulate_events_dataset(num_users=2, days=1, seed=1, price_beta=7.5)
    assert seen["price_beta"] == 7.5


def test_reload_duckdb_warehouse_loads_only_raw_screens(tmp_path, small):
    events, profiles = small
    db_file = tmp_path / "sim.duckdb"
    counts = reload_duckdb_warehouse(events[:150], profiles, db_path=str(db_file), workers=1)
    assert counts["raw_events"] == 150
    assert counts["fact_events"] == 150
    assert 0 < counts["dim_users"] <= 12
    con = duckdb.connect(str(db_file), read_only=True)
    try:
        bad = con.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE column_name = 'archetype' OR column_name LIKE 'true\\_%' ESCAPE '\\' "
            "OR column_name IN ('state', 'intent_family', 'is_deliberate_comparison')"
        ).fetchall()
    finally:
        con.close()
    assert bad == []


def test_v1_engine_is_gone():
    source = Path(simulate_users.__file__).read_text()
    assert "ARCHETYPE_PARAMS" not in source
    assert "_get_catalog_extraction" not in source
    assert "import random" not in source
    assert "synthetic_data import CATALOG" not in source
    assert "build_catalog" in source

"""Tests for the simulator calibration against RetailRocket (S3)."""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import pytest

from src.simulation.calibrate import (
    DELIBERATE_DIFFERENCES,
    QUANTITIES,
    build_calibration,
    calibration_stats,
)


def _warehouse(path, rows):
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE fact_events (event_id VARCHAR, user_id VARCHAR, timestamp TIMESTAMP, "
                "action VARCHAR, session_id VARCHAR)")
    con.executemany("INSERT INTO fact_events VALUES (?, ?, ?, ?, ?)", rows)
    con.close()
    return path


T0 = datetime(2026, 10, 1, 12)
# u1: one session of 3 (view, cart, buy) on day 0 and a single view on day 1.
# u2: one session of 2 views on day 0.
ROWS = [
    ("e1", "u1", T0, "view", "s1"), ("e2", "u1", T0 + timedelta(minutes=1), "cart", "s1"),
    ("e3", "u1", T0 + timedelta(minutes=2), "buy", "s1"), ("e4", "u1", T0 + timedelta(days=1), "view", "s2"),
    ("e5", "u2", T0, "view", "s3"), ("e6", "u2", T0 + timedelta(minutes=3), "view", "s3"),
]


def test_calibration_stats_hand_computed(tmp_path):
    con = duckdb.connect(str(_warehouse(tmp_path / "w.duckdb", ROWS)), read_only=True)
    stats = calibration_stats(con)
    con.close()
    assert set(stats) == set(QUANTITIES)
    # u1: 4 events over 2 active days = 2.0; u2: 2 events over 1 day = 2.0.
    assert stats["events_per_user_per_active_day"] == pytest.approx(2.0)
    assert stats["median_events_per_session"] == pytest.approx(2.0)
    assert stats["single_event_session_share"] == pytest.approx(1 / 3, abs=1e-4)
    assert stats["cart_to_view_ratio"] == pytest.approx(1 / 4, abs=1e-4)
    assert stats["buy_to_cart_ratio"] == pytest.approx(1.0)
    assert stats["share_users_with_purchase"] == pytest.approx(0.5)


def test_ratios_are_none_without_denominator(tmp_path):
    rows = [("e1", "u1", T0, "view", "s1")]
    con = duckdb.connect(str(_warehouse(tmp_path / "w.duckdb", rows)), read_only=True)
    stats = calibration_stats(con)
    con.close()
    assert stats["buy_to_cart_ratio"] is None
    assert stats["cart_to_view_ratio"] == 0.0


def test_build_calibration_has_both_columns_and_differences(tmp_path):
    sim = _warehouse(tmp_path / "sim.duckdb", ROWS)
    rr = _warehouse(tmp_path / "rr.duckdb", ROWS[:3])
    out = build_calibration(sim, rr)
    for q in QUANTITIES:
        assert set(out["quantities"][q]) == {"simulated", "retailrocket"}
    assert out["deliberate_differences"] and all(isinstance(s, str) and s for s in out["deliberate_differences"])
    assert out["deliberate_differences"] == DELIBERATE_DIFFERENCES


def test_build_calibration_opens_read_only(tmp_path):
    sim = _warehouse(tmp_path / "sim.duckdb", ROWS)
    before = sim.stat().st_mtime_ns
    build_calibration(sim, sim)
    assert sim.stat().st_mtime_ns == before

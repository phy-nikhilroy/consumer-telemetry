"""Tests for the price effect experiment: informed-purchase frame, regression recovery, isolation and a tiny run."""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

from src.config import DEFAULT_DB_PATH
from src.experiments import price_effect as pe


def _db(path: Path, events: list, obs: list) -> None:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE fact_events (event_id VARCHAR, user_id VARCHAR, timestamp TIMESTAMP, "
                "app_name VARCHAR, canonical_product_id VARCHAR, action VARCHAR)")
    con.execute("CREATE TABLE fact_price_observations (observation_id VARCHAR, source_event_id VARCHAR, "
                "user_id VARCHAR, timestamp TIMESTAMP, app_name VARCHAR, canonical_product_id VARCHAR, "
                "price_inr DOUBLE)")
    for e in events:
        con.execute("INSERT INTO fact_events VALUES (?,?,?,?,?,?)", e)
    for o in obs:
        con.execute("INSERT INTO fact_price_observations VALUES (?,?,?,?,?,?,?)", o)
    con.close()


def test_informed_purchase_frame_picks_later_and_earlier_view(tmp_path: Path) -> None:
    events = [("v1", "u", "2026-10-01 10:00:00", "a", "P", "view"),
              ("v2", "u", "2026-10-01 11:00:00", "b", "P", "view"),
              ("b1", "u", "2026-10-01 12:00:00", "b", "P", "buy"),
              ("x1", "u", "2026-10-01 13:00:00", "a", "Q", "view")]
    obs = [("v1#0", "v1", "u", "2026-10-01 10:00:00", "a", "P", 100.0),
           ("v2#0", "v2", "u", "2026-10-01 11:00:00", "b", "P", 150.0),
           ("b1#0", "b1", "u", "2026-10-01 12:00:00", "b", "P", 150.0)]
    _db(tmp_path / "w.duckdb", events, obs)
    f = pe.informed_purchase_frame(tmp_path / "w.duckdb")
    assert f.height == 1
    row = f.row(0, named=True)
    assert row["y"] == 1 and row["later_is_dominant"] == 0
    assert row["log_price_ratio"] == pytest.approx(math.log(1.5))


def test_informed_purchase_frame_ignores_uninformed_and_stale(tmp_path: Path) -> None:
    events = [("v1", "u", "2026-10-01 10:00:00", "a", "P", "view"),
              ("b1", "u", "2026-10-05 12:00:00", "b", "P", "buy"),
              ("b2", "u", "2026-10-06 12:00:00", "a", "P", "buy")]
    obs = [("v1#0", "v1", "u", "2026-10-01 10:00:00", "a", "P", 100.0)]
    _db(tmp_path / "w.duckdb", events, obs)
    assert pe.informed_purchase_frame(tmp_path / "w.duckdb").height == 0


def test_regression_recovers_planted_coefficient() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(0, 0.3, n)
    dom = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(-3.0 * x + 0.5 * dom)))
    y = (rng.random(n) < p).astype(int)
    frame = pl.DataFrame({"user_id": [f"u{i % 400}" for i in range(n)], "y": y,
                          "log_price_ratio": x, "later_is_dominant": dom})
    out = pe.price_ratio_regression(frame, n_boot=100)
    assert out["n"] == n
    assert out["coef_log_price_ratio"] == pytest.approx(-3.0, abs=0.6)
    assert out["ci"]["lo"] < -3.0 < out["ci"]["hi"] or abs(out["coef_log_price_ratio"] + 3.0) < 0.6
    assert out["ci"]["lo"] <= out["coef_log_price_ratio"] <= out["ci"]["hi"]


def test_regression_degenerate_inputs_give_nan() -> None:
    empty = pl.DataFrame(schema={"user_id": pl.String, "y": pl.Int8, "log_price_ratio": pl.Float64,
                                 "later_is_dominant": pl.Int8})
    assert math.isnan(pe.price_ratio_regression(empty)["coef_log_price_ratio"])
    one = pl.DataFrame({"user_id": ["a", "b"], "y": [1, 1], "log_price_ratio": [0.1, 0.2],
                        "later_is_dominant": [0, 1]})
    out = pe.price_ratio_regression(one)
    assert math.isnan(out["coef_log_price_ratio"]) and out["n"] == 2


def test_defaults_match_spec_and_isolated() -> None:
    assert pe.PRICE_BETAS == (0.0, 4.0, 8.0) and len(pe.SEEDS) == 3
    assert pe.USERS == 500 and pe.DAYS == 45
    assert "data/experiments" in str(pe.DEFAULT_OUT)
    assert not str(pe.DEFAULT_OUT).startswith(str(DEFAULT_DB_PATH.parent))


def test_tiny_end_to_end(tmp_path: Path) -> None:
    res = pe.run_price_effect(tmp_path, betas=(0.0, 8.0), seeds=(1,), users=40, days=20, n_boot=20)
    assert set(res["levels"]) == {"0", "8"}
    for lvl in res["levels"].values():
        assert "mean" in lvl["chose_cheaper_by_seed"][0]
        assert "ci" in lvl["regression"]

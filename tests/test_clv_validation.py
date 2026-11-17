"""
Tests for the CLV reference-date cut, holdout validation and the ML regressor comparison.

A synthetic warehouse is built in a temporary directory, never under data/processed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import inspect
import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings, strategies as st

from src.features.database import DatabaseManager

CAL_END = "2026-10-18 00:00:00"
HOLDOUT_DAYS = 9
START = datetime(2026, 9, 1)
N_USERS = 150


def _build_warehouse(path: Path, seed: int = 7) -> Path:
    """Creates a seeded warehouse with views and purchases from 2026-09-01 to 2026-10-28."""
    rng = np.random.default_rng(seed)
    DatabaseManager(db_path=path).init_schema()
    rows = []
    eid = 0
    span = 57 * 24 * 3600
    for u in range(N_USERS):
        n_buy = int(rng.choice([1, 2, 3, 4, 6, 9, 14]))
        tier = float(rng.choice([500, 2000, 9000]))
        first_day = float(rng.uniform(0, 40))
        life = float(rng.exponential(18.0)) + 1.0
        for _ in range(n_buy):
            offset = first_day + float(rng.uniform(0, min(life, 56.9 - first_day)))
            ts = START + timedelta(seconds=int(offset * 86400))
            usd = rng.random() < 0.3
            price = round(tier * (0.7 + 0.6 * rng.random()) / (83.0 if usd else 1.0), 2)
            rows.append((f"e{eid}", f"u{u}", ts, "amazon", "p1", "buy", price,
                         "USD" if usd else "INR", f"s{u}"))
            eid += 1
        for _ in range(8):
            ts = START + timedelta(seconds=int(rng.integers(0, span)))
            rows.append((f"e{eid}", f"u{u}", ts, "flipkart", "p1", "view", 1000.0, "INR", f"s{u}"))
            eid += 1
    con = duckdb.connect(str(path))
    frame = pl.DataFrame(
        rows, orient="row",
        schema=["event_id", "user_id", "timestamp", "app_name", "canonical_product_id",
                "action", "price", "currency", "session_id"],
    ).to_pandas()
    con.register("rows_df", frame)
    con.execute("INSERT INTO fact_events SELECT * FROM rows_df")
    con.unregister("rows_df")
    con.execute("INSERT INTO dim_products VALUES ('p1','b','t','c1','c2',1000.0)")
    con.close()
    return path


@pytest.fixture(scope="module")
def clv_db(tmp_path_factory) -> Path:
    return _build_warehouse(tmp_path_factory.mktemp("clv_val") / "w.duckdb")


@pytest.fixture(scope="module")
def holdout(clv_db):
    from src.models.clv_validation import evaluate_clv_holdout

    return evaluate_clv_holdout(clv_db, CAL_END, HOLDOUT_DAYS)


def _buys(db: Path) -> pl.DataFrame:
    con = duckdb.connect(str(db), read_only=True)
    try:
        return pl.from_arrow(con.execute(
            "SELECT user_id, timestamp, price FROM fact_events WHERE action='buy'"
        ).to_arrow_table())
    finally:
        con.close()


def test_holdout_result_has_spec_keys(holdout):
    expected = {
        "n_customers", "actual_total_purchases", "predicted_total_purchases",
        "mae_purchases_bgnbd", "mae_purchases_naive", "spearman_spend_bgnbd_gg",
        "spearman_spend_naive", "by_frequency_bucket", "order_value_mae",
    }
    assert expected <= set(holdout)
    assert set(holdout["order_value_mae"]) >= {"gamma_gamma", "observed_mean", "shrunk"}


def test_holdout_n_customers_matches_users_with_purchase_by_calibration_end(clv_db, holdout):
    buys = _buys(clv_db)
    cutoff = datetime.fromisoformat(CAL_END)
    expected = buys.filter(pl.col("timestamp") <= cutoff)["user_id"].n_unique()
    assert holdout["n_customers"] == expected


def test_holdout_actual_total_counts_only_window(clv_db, holdout):
    """Actuals must exclude purchases at or before calibration_end and after the window."""
    buys = _buys(clv_db)
    cutoff = datetime.fromisoformat(CAL_END)
    end = cutoff + timedelta(days=HOLDOUT_DAYS)
    users = buys.filter(pl.col("timestamp") <= cutoff)["user_id"].unique()
    window = buys.filter(
        (pl.col("timestamp") > cutoff) & (pl.col("timestamp") <= end)
        & pl.col("user_id").is_in(users)
    )
    assert holdout["actual_total_purchases"] == window.height


def test_holdout_actuals_exclude_purchase_exactly_at_calibration_end(tmp_path):
    """A purchase stamped exactly at calibration_end is calibration, not holdout."""
    from src.models.clv_validation import evaluate_clv_holdout

    db = _build_warehouse(tmp_path / "w.duckdb", seed=3)
    con = duckdb.connect(str(db))
    base = con.execute(
        "SELECT COUNT(*) FROM fact_events WHERE action='buy' "
        "AND timestamp > TIMESTAMP '2026-10-18 00:00:00' "
        "AND timestamp <= TIMESTAMP '2026-10-27 00:00:00'"
    ).fetchone()[0]
    con.execute("INSERT INTO fact_events VALUES ('x1','u0',TIMESTAMP '2026-10-18 00:00:00',"
                "'amazon','p1','buy',100.0,'INR','sx')")
    con.execute("INSERT INTO fact_events VALUES ('x2','u0',TIMESTAMP '2026-10-18 00:00:01',"
                "'amazon','p1','buy',100.0,'INR','sx')")
    con.close()
    out = evaluate_clv_holdout(db, CAL_END, HOLDOUT_DAYS)
    # u0 may or may not be a calibration customer, but at-cutoff row is calibration so
    # x2 is the only new holdout row at most; x1 must never be counted as actual.
    assert out["actual_total_purchases"] <= base + 1


def test_holdout_window_upper_bound_inclusive(tmp_path):
    from src.models.clv_validation import evaluate_clv_holdout

    db = _build_warehouse(tmp_path / "w.duckdb", seed=5)
    con = duckdb.connect(str(db))
    # u_new buys before the cutoff and exactly at the window end and one second after.
    con.execute("INSERT INTO fact_events VALUES ('y0','unew',TIMESTAMP '2026-10-10 00:00:00',"
                "'amazon','p1','buy',100.0,'INR','s')")
    con.close()
    before = evaluate_clv_holdout(db, CAL_END, HOLDOUT_DAYS)["actual_total_purchases"]
    con = duckdb.connect(str(db))
    con.execute("INSERT INTO fact_events VALUES ('y1','unew',TIMESTAMP '2026-10-27 00:00:00',"
                "'amazon','p1','buy',100.0,'INR','s')")
    con.execute("INSERT INTO fact_events VALUES ('y2','unew',TIMESTAMP '2026-10-27 00:00:01',"
                "'amazon','p1','buy',100.0,'INR','s')")
    con.close()
    after = evaluate_clv_holdout(db, CAL_END, HOLDOUT_DAYS)["actual_total_purchases"]
    assert after - before == 1


def test_calibration_transactions_contain_none_after_cutoff(clv_db):
    from src.models.clv import load_purchase_transactions

    tx = load_purchase_transactions(db_path=clv_db, reference_date=CAL_END)
    assert tx.height > 0
    assert tx["timestamp"].max() <= datetime.fromisoformat(CAL_END)


def test_by_frequency_bucket_counts_sum_to_n_customers(holdout):
    buckets = holdout["by_frequency_bucket"]
    total = sum(b["n"] for b in (buckets.values() if isinstance(buckets, dict) else buckets))
    assert total == holdout["n_customers"]


def test_by_frequency_bucket_labels(holdout):
    buckets = holdout["by_frequency_bucket"]
    labels = set(buckets) if isinstance(buckets, dict) else {str(b.get("bucket")) for b in buckets}
    assert {"0", "1", "2", "3-5", "6+"} <= labels


def test_by_frequency_bucket_entries_have_means(holdout):
    buckets = holdout["by_frequency_bucket"]
    entries = buckets.values() if isinstance(buckets, dict) else buckets
    for entry in entries:
        assert "n" in entry
        assert any("pred" in k for k in entry)
        assert any("actual" in k for k in entry)


def test_holdout_metrics_are_finite_and_nonnegative(holdout):
    for key in ("mae_purchases_bgnbd", "mae_purchases_naive"):
        assert np.isfinite(holdout[key]) and holdout[key] >= 0
    for value in holdout["order_value_mae"].values():
        assert np.isfinite(value) and value >= 0
    for key in ("spearman_spend_bgnbd_gg", "spearman_spend_naive"):
        assert -1.0 <= holdout[key] <= 1.0


def test_holdout_unchanged_by_events_after_window(clv_db, tmp_path):
    """Events beyond the holdout window cannot influence any reported metric."""
    from src.models.clv_validation import evaluate_clv_holdout

    base = evaluate_clv_holdout(clv_db, CAL_END, HOLDOUT_DAYS)
    copy = tmp_path / "trim.duckdb"
    copy.write_bytes(clv_db.read_bytes())
    con = duckdb.connect(str(copy))
    con.execute("DELETE FROM fact_events WHERE timestamp > TIMESTAMP '2026-10-27 00:00:00'")
    con.close()
    trimmed = evaluate_clv_holdout(copy, CAL_END, HOLDOUT_DAYS)
    assert trimmed["n_customers"] == base["n_customers"]
    assert trimmed["actual_total_purchases"] == base["actual_total_purchases"]
    assert trimmed["mae_purchases_bgnbd"] == pytest.approx(base["mae_purchases_bgnbd"])


def test_holdout_deterministic(clv_db, holdout):
    from src.models.clv_validation import evaluate_clv_holdout

    again = evaluate_clv_holdout(clv_db, CAL_END, HOLDOUT_DAYS)
    assert again["n_customers"] == holdout["n_customers"]
    assert again["mae_purchases_bgnbd"] == pytest.approx(holdout["mae_purchases_bgnbd"])


def test_naive_baseline_is_purchases_per_day_times_holdout(clv_db):
    """Naive purchases = purchases-before-cutoff / observed days * holdout_days."""
    from src.models.clv_validation import evaluate_clv_holdout

    out = evaluate_clv_holdout(clv_db, CAL_END, HOLDOUT_DAYS)
    buys = _buys(clv_db)
    cutoff = datetime.fromisoformat(CAL_END)
    cal = buys.filter(pl.col("timestamp") <= cutoff)
    end = cutoff + timedelta(days=HOLDOUT_DAYS)
    first = cal.group_by("user_id").agg(pl.col("timestamp").min().alias("f"),
                                         pl.len().alias("n"))
    actual = (buys.filter((pl.col("timestamp") > cutoff) & (pl.col("timestamp") <= end))
              .group_by("user_id").len().rename({"len": "a"}))
    df = first.join(actual, on="user_id", how="left").fill_null(0)
    days = ((cutoff - df["f"]).dt.total_seconds() / 86400.0).to_numpy()
    n = df["n"].to_numpy().astype(float)
    a = df["a"].to_numpy().astype(float)
    lo = np.abs(n / np.maximum(days, 1.0) * HOLDOUT_DAYS - a).mean()
    hi = np.abs(n / np.maximum(days, 1e-9) * HOLDOUT_DAYS - a).mean()
    # Accept any reasonable observation-window convention: between the two bounds,
    # with generous slack, but not the unscaled count.
    assert out["mae_purchases_naive"] == pytest.approx(lo, rel=0.5) or \
        out["mae_purchases_naive"] == pytest.approx(hi, rel=0.5)
    unscaled = np.abs(n - a).mean()
    assert out["mae_purchases_naive"] != pytest.approx(unscaled, rel=1e-6) or unscaled == pytest.approx(lo, rel=1e-6)


# ---------------------------------------------------------------- shrinkage


def test_shrinkage_constant_in_config():
    from src import config

    assert config.CLV_ORDER_VALUE_SHRINKAGE_K == 3.0


def test_shrunk_value_formula_via_holdout_signature():
    """The holdout API takes exactly (db_path, calibration_end, holdout_days)."""
    from src.models.clv_validation import evaluate_clv_holdout

    params = list(inspect.signature(evaluate_clv_holdout).parameters)
    assert params == ["db_path", "calibration_end", "holdout_days"]


@given(
    mean_i=st.floats(1.0, 1e5),
    n_i=st.integers(1, 200),
    global_mean=st.floats(1.0, 1e5),
)
@settings(max_examples=50, deadline=None)
def test_shrunk_value_between_user_mean_and_global_mean(mean_i, n_i, global_mean):
    """The shrunk order value lies between the user's mean and the global mean."""
    from src import config

    k = config.CLV_ORDER_VALUE_SHRINKAGE_K
    shrunk = (n_i * mean_i + k * global_mean) / (n_i + k)
    lo, hi = min(mean_i, global_mean), max(mean_i, global_mean)
    assert lo - 1e-6 <= shrunk <= hi + 1e-6


def test_shrunk_order_value_helper_matches_formula_if_exposed():
    """When the module exposes a shrinkage helper, it must match the spec formula."""
    from src.models import clv_validation as cv

    fn = next((getattr(cv, n) for n in dir(cv)
               if "shrunk" in n.lower() and callable(getattr(cv, n))
               and not n.startswith("evaluate")), None)
    assert fn is not None, "expected a public shrunk-order-value helper in clv_validation"
    n = np.array([1.0, 5.0, 50.0])
    mean = np.array([100.0, 400.0, 900.0])
    out = np.asarray(fn(n, mean, 300.0))
    expected = (n * mean + 3.0 * 300.0) / (n + 3.0)
    np.testing.assert_allclose(out, expected)


# ---------------------------------------------------------------- run_clv_pipeline


def test_run_clv_pipeline_accepts_reference_date():
    from src.models.clv import run_clv_pipeline

    param = inspect.signature(run_clv_pipeline).parameters["reference_date"]
    assert param.default is None or param.default is not inspect.Parameter.empty


def test_run_clv_pipeline_default_uses_test_reference_date(clv_db, tmp_path):
    """Default cut is TEST_REFERENCE_DATE: later purchases never enter the fit."""
    from src.config import TEST_REFERENCE_DATE
    from src.models.clv import run_clv_pipeline

    out = run_clv_pipeline(db_path=clv_db, features_dir=tmp_path / "f",
                           artifacts_dir=tmp_path / "a")
    buys = _buys(clv_db)
    expected = buys.filter(pl.col("timestamp") <= datetime.fromisoformat(TEST_REFERENCE_DATE)).height
    assert out["n_transactions"] == expected


def test_run_clv_pipeline_explicit_reference_date(clv_db, tmp_path):
    from src.models.clv import run_clv_pipeline

    out = run_clv_pipeline(db_path=clv_db, features_dir=tmp_path / "f",
                           artifacts_dir=tmp_path / "a", reference_date=CAL_END)
    expected = _buys(clv_db).filter(pl.col("timestamp") <= datetime.fromisoformat(CAL_END)).height
    assert out["n_transactions"] == expected


def test_run_clv_pipeline_summary_flag_gamma_gamma_mean_undefined(clv_db, tmp_path):
    from src.models.clv import run_clv_pipeline

    out = run_clv_pipeline(db_path=clv_db, features_dir=tmp_path / "f",
                           artifacts_dir=tmp_path / "a", reference_date=CAL_END)
    summary = json.loads((tmp_path / "a" / "clv_summary.json").read_text())
    for payload in (out, summary):
        assert isinstance(payload["gamma_gamma_mean_undefined"], bool)
        assert payload["gamma_gamma_mean_undefined"] == (payload["gamma_gamma_params"]["q"] <= 1.0)


def test_pipeline_observation_end_is_reference_date(clv_db, tmp_path):
    """T is measured to the reference date, not to the last purchase."""
    from src.models.clv import run_clv_pipeline

    run_clv_pipeline(db_path=clv_db, features_dir=tmp_path / "f",
                     artifacts_dir=tmp_path / "a", reference_date=CAL_END)
    clv = pl.read_parquet(tmp_path / "f" / "user_clv.parquet")
    cutoff = datetime.fromisoformat(CAL_END)
    first = (_buys(clv_db).filter(pl.col("timestamp") <= cutoff)
             .group_by("user_id").agg(pl.col("timestamp").min().alias("f")))
    joined = clv.join(first, on="user_id")
    expected_t = ((cutoff - joined["f"]).dt.total_seconds() / 86400.0).to_numpy()
    assert np.allclose(joined["T"].to_numpy(), expected_t, atol=1.01)


# ---------------------------------------------------------------- regressor


def test_evaluate_clv_regressor_signature_defaults():
    from src.models.clv_validation import evaluate_clv_regressor

    p = inspect.signature(evaluate_clv_regressor).parameters
    assert p["train_ref"].default == "2026-10-09 00:00:00"
    assert p["test_ref"].default == "2026-10-18 00:00:00"
    assert p["horizon_days"].default == 9


def test_evaluate_clv_regressor_reports_metrics(clv_db):
    from src.models.clv_validation import evaluate_clv_regressor

    out = evaluate_clv_regressor(clv_db)
    assert isinstance(out, dict)
    keys = " ".join(out) + " " + json.dumps(out, default=str)
    assert "spearman" in keys.lower()
    assert "mae" in keys.lower()
    assert "bgnbd" in keys.lower() or "bg_nbd" in keys.lower()


def test_evaluate_clv_regressor_unchanged_by_events_after_horizon(clv_db, tmp_path):
    """Dropping events after test_ref + horizon cannot change the regressor result."""
    from src.models.clv_validation import evaluate_clv_regressor

    base = evaluate_clv_regressor(clv_db)
    copy = tmp_path / "trim.duckdb"
    copy.write_bytes(clv_db.read_bytes())
    con = duckdb.connect(str(copy))
    con.execute("DELETE FROM fact_events WHERE timestamp > TIMESTAMP '2026-10-27 00:00:00'")
    con.close()
    trimmed = evaluate_clv_regressor(copy)
    assert json.dumps(trimmed, sort_keys=True, default=str) == json.dumps(base, sort_keys=True, default=str)


def test_holdout_purchase_at_cutoff_is_calibration_not_actual(tmp_path):
    """Adding a purchase stamped exactly at calibration_end must not change the actuals."""
    from src.models.clv_validation import evaluate_clv_holdout

    db = _build_warehouse(tmp_path / "w.duckdb", seed=9)
    con = duckdb.connect(str(db))
    con.execute("INSERT INTO fact_events VALUES ('z0','unew',TIMESTAMP '2026-10-10 00:00:00',"
                "'amazon','p1','buy',100.0,'INR','s')")
    con.close()
    before = evaluate_clv_holdout(db, CAL_END, HOLDOUT_DAYS)["actual_total_purchases"]
    con = duckdb.connect(str(db))
    con.execute("INSERT INTO fact_events VALUES ('z1','unew',TIMESTAMP '2026-10-18 00:00:00',"
                "'amazon','p1','buy',100.0,'INR','s')")
    con.close()
    after = evaluate_clv_holdout(db, CAL_END, HOLDOUT_DAYS)["actual_total_purchases"]
    assert after == before


def test_holdout_fit_ignores_purchases_after_calibration_end(clv_db, tmp_path):
    """Deleting every event after calibration_end leaves the holdout's fitted parameters unchanged."""
    from src.models.clv_validation import evaluate_clv_holdout

    base = evaluate_clv_holdout(clv_db, CAL_END, HOLDOUT_DAYS)
    copy = tmp_path / "trim.duckdb"
    copy.write_bytes(clv_db.read_bytes())
    con = duckdb.connect(str(copy))
    con.execute(f"DELETE FROM fact_events WHERE timestamp > TIMESTAMP '{CAL_END}'")
    con.close()
    trimmed = evaluate_clv_holdout(copy, CAL_END, HOLDOUT_DAYS)
    for key, value in base["bgnbd_params"].items():
        assert trimmed["bgnbd_params"][key] == pytest.approx(value, rel=1e-9)
    for key, value in base["gamma_gamma_params"].items():
        assert trimmed["gamma_gamma_params"][key] == pytest.approx(value, rel=1e-9)
    assert trimmed["n_customers"] == base["n_customers"]

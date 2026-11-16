"""
Unit tests for BG/NBD and Gamma-Gamma customer lifetime value modeling.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.config import USD_TO_INR_RATE
from src.models.clv import (
    build_rfm_summary,
    fit_bgnbd,
    fit_gamma_gamma,
    load_purchase_transactions,
    load_user_archetypes,
    observed_order_values,
    plot_clv_distribution,
    predict_clv,
    summarize_archetype_clv,
)

DB_PATH = Path("data/processed/consumer.duckdb")


def _require_warehouse():
    """Skips a test when the warehouse has not been built."""
    if not DB_PATH.exists():
        pytest.skip("Warehouse database not available")


@pytest.fixture(scope="module")
def transactions():
    """Loads the purchase transaction log once for the whole module."""
    _require_warehouse()
    return load_purchase_transactions(db_path=DB_PATH)


@pytest.fixture(scope="module")
def summary(transactions):
    """Builds the per-customer RFM summary once."""
    return build_rfm_summary(transactions)


@pytest.fixture(scope="module")
def fitted(summary):
    """Fits both CLV models once; fitting is the slow part of this module."""
    bgf = fit_bgnbd(summary)
    ggf, repeat = fit_gamma_gamma(summary)
    return bgf, ggf, repeat


def test_load_purchase_transactions_normalizes_currency(transactions):
    """Verifies only purchases are returned and revenue is normalized to INR."""
    assert transactions.height > 0
    for col in ("user_id", "timestamp", "revenue_inr"):
        assert col in transactions.columns

    revenue = transactions["revenue_inr"]
    assert revenue.min() > 0.0
    assert revenue.is_finite().all()

    # The cheapest catalog item lists at 1,099 INR and the deepest discount is 50
    # percent, so no converted purchase can fall below about 550 rupees. An
    # unnormalized mix would leave raw dollar amounts in the low tens sitting
    # alongside rupee amounts in the thousands, so a floor well above 100 confirms
    # the conversion was applied.
    assert revenue.min() > 100.0
    # USD prices run from about 20 to 1,300 dollars, which after conversion must all
    # land in the same rupee scale as the INR listings.
    assert revenue.max() > 1_000.0


def test_build_rfm_summary_follows_lifetimes_conventions(summary):
    """Verifies the summary columns and the repeat-purchase frequency convention."""
    for col in ("frequency", "recency", "T", "monetary_value"):
        assert col in summary.columns

    assert len(summary) > 0
    assert (summary["frequency"] >= 0).all()
    # Recency can never exceed the customer's observed age.
    assert (summary["recency"] <= summary["T"] + 1e-9).all()
    # frequency counts repeat purchases, so a one-off buyer scores 0 and carries no
    # repeat monetary value.
    one_off = summary[summary["frequency"] == 0]
    if len(one_off) > 0:
        assert (one_off["monetary_value"] == 0).all()


def test_observed_order_values_covers_every_buyer(transactions, summary):
    """Verifies the observed fallback is available for all customers in the summary."""
    observed = observed_order_values(transactions)

    assert observed.height == len(summary)
    assert observed["observed_avg_order_value"].min() > 0.0
    assert observed["observed_purchase_count"].min() >= 1

    # Every summary customer must have an observed value, since the summary is derived
    # from the same transaction log.
    assert set(observed["user_id"].to_list()) == set(summary.index.astype(str).tolist())


def test_bgnbd_dropout_parameters_are_identified(fitted):
    """Verifies the fitted dropout process is non-degenerate."""
    bgf, _, _ = fitted
    params = {k: float(v) for k, v in bgf.params_.items()}

    for name in ("r", "alpha", "a", "b"):
        assert name in params
        assert np.isfinite(params[name])
        assert params[name] > 0

    # A dropout parameter collapsing toward zero means the model found no churn signal
    # and P(alive) degenerates to 1.0 for every customer, which makes the expected
    # purchase formula divide by zero.
    assert params["a"] > 1e-4


def test_probability_alive_varies_across_customers(fitted, summary):
    """Verifies P(alive) spans a real range rather than collapsing to a constant."""
    bgf, _, _ = fitted
    alive = np.asarray(
        bgf.conditional_probability_alive(
            summary["frequency"], summary["recency"], summary["T"]
        ),
        dtype=float,
    )

    assert np.isfinite(alive).all()
    assert ((alive >= 0.0) & (alive <= 1.0)).all()
    assert alive.max() - alive.min() > 0.1


def test_gamma_gamma_fitted_on_repeat_customers_only(fitted, summary):
    """Verifies the monetary model excludes one-off and zero-value customers."""
    _, ggf, repeat = fitted

    assert (repeat["frequency"] > 0).all()
    assert (repeat["monetary_value"] > 0).all()
    assert len(repeat) <= len(summary)

    for name in ("p", "q", "v"):
        assert np.isfinite(float(ggf.params_[name]))
        assert float(ggf.params_[name]) > 0


def test_gamma_gamma_raises_without_repeat_customers():
    """Verifies a clear error rather than a degenerate fit when no repeats exist."""
    import pandas as pd

    only_one_off = pd.DataFrame(
        {
            "frequency": [0.0, 0.0],
            "recency": [0.0, 0.0],
            "T": [10.0, 12.0],
            "monetary_value": [0.0, 0.0],
        },
        index=["user_a", "user_b"],
    )
    with pytest.raises(ValueError, match="repeat customers"):
        fit_gamma_gamma(only_one_off)


def test_build_rfm_summary_rejects_empty_log():
    """Verifies an explicit error for an empty transaction log."""
    empty = pl.DataFrame(
        {"user_id": [], "timestamp": [], "revenue_inr": []},
        schema={"user_id": pl.String, "timestamp": pl.Datetime, "revenue_inr": pl.Float64},
    )
    with pytest.raises(ValueError, match="empty transaction log"):
        build_rfm_summary(empty)


def test_predict_clv_produces_usable_values(fitted, summary, transactions):
    """Verifies every CLV prediction is finite, non-negative, and internally consistent."""
    bgf, ggf, _ = fitted
    clv = predict_clv(bgf, ggf, summary, transactions, horizon_days=30)

    assert clv.height == len(summary)
    assert clv["predicted_clv"].is_finite().all()
    assert clv["predicted_clv"].min() >= 0.0
    assert clv["predicted_purchases"].min() >= 0.0
    assert clv["predicted_avg_order_value"].min() > 0.0

    # CLV must be exactly the product the specification defines.
    product = clv["predicted_purchases"] * clv["predicted_avg_order_value"]
    np.testing.assert_allclose(
        clv["predicted_clv"].to_numpy(), product.to_numpy(), rtol=1e-9
    )

    # Output must be sorted by CLV descending.
    values = clv["predicted_clv"].to_list()
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_predict_clv_records_estimate_provenance(fitted, summary, transactions):
    """Verifies each row states whether Gamma-Gamma or the observed fallback was used."""
    bgf, ggf, _ = fitted
    clv = predict_clv(bgf, ggf, summary, transactions, horizon_days=30)

    assert set(clv["order_value_estimate_source"].unique().to_list()).issubset(
        {"gamma_gamma", "observed"}
    )
    assert set(clv["purchase_estimate_source"].unique().to_list()).issubset(
        {"bg_nbd", "observed_rate"}
    )

    # One-off buyers cannot use Gamma-Gamma, so they must fall back.
    one_off = clv.filter(pl.col("frequency") == 0)
    if one_off.height > 0:
        assert (one_off["order_value_estimate_source"] == "observed").all()

    # Repeat customers should overwhelmingly use the fitted monetary model.
    repeat = clv.filter(pl.col("frequency") > 0)
    gg_share = (repeat["order_value_estimate_source"] == "gamma_gamma").mean()
    assert gg_share > 0.9


def test_clv_horizon_scales_predicted_purchases(fitted, summary, transactions):
    """Verifies a longer horizon predicts more purchases for the same customers."""
    bgf, ggf, _ = fitted
    short = predict_clv(bgf, ggf, summary, transactions, horizon_days=7).sort("user_id")
    long = predict_clv(bgf, ggf, summary, transactions, horizon_days=60).sort("user_id")

    assert (
        long["predicted_purchases"].to_numpy() >= short["predicted_purchases"].to_numpy() - 1e-9
    ).all()
    assert long["predicted_purchases"].sum() > short["predicted_purchases"].sum()


def test_clv_recovers_simulated_archetype_ordering(fitted, summary, transactions):
    """
    Verifies predicted CLV ranks the simulated archetypes in the expected order.

    The archetype is simulator ground truth and is never a model feature, so this is an
    independent check that the model recovers the latent behavior that generated the data.
    """
    archetypes = load_user_archetypes()
    if archetypes is None:
        pytest.skip("Simulated event log with archetypes not available")

    bgf, ggf, _ = fitted
    clv = predict_clv(bgf, ggf, summary, transactions, horizon_days=30).join(
        archetypes, on="user_id", how="left"
    )
    profile = summarize_archetype_clv(clv)
    assert profile is not None

    means = dict(zip(profile["archetype"].to_list(), profile["mean_predicted_clv"].to_list()))

    # Loyalists were generated with the highest purchase propensity and window shoppers
    # with the lowest, so their predicted value must separate accordingly.
    assert means["loyalist"] > means["comparison_shopper"]
    assert means["loyalist"] > means["window_shopper"]
    assert means["impulse_buyer"] > means["window_shopper"]


def test_plot_clv_distribution_writes_image(fitted, summary, transactions, tmp_path):
    """Verifies the CLV diagnostics figure is written to disk."""
    bgf, ggf, _ = fitted
    clv = predict_clv(bgf, ggf, summary, transactions, horizon_days=30)

    out = plot_clv_distribution(clv, output_path=tmp_path / "clv.png")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_load_purchase_transactions_filters_users_and_date(tiny_warehouse):
    """The serving path loads one user's purchases up to a reference date."""
    import duckdb

    con = duckdb.connect(str(tiny_warehouse), read_only=True)
    try:
        everything = load_purchase_transactions(con=con)
        user = everything["user_id"][0]
        cutoff = everything.filter(pl.col("user_id") == user)["timestamp"].sort()[0]
        one_user = load_purchase_transactions(con=con, user_ids=[user])
        up_to_first = load_purchase_transactions(con=con, user_ids=[user], reference_date=cutoff)
    finally:
        con.close()
    assert one_user.height == everything.filter(pl.col("user_id") == user).height
    assert set(one_user["user_id"]) == {user}
    assert up_to_first.height >= 1
    assert up_to_first["timestamp"].max() <= cutoff

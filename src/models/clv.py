"""
Customer lifetime value modeling with BG/NBD and Gamma-Gamma.

Two separate questions are modeled and then multiplied together:

  BG/NBD       how many purchases will this user make over the horizon, and are they
               still an active customer at all?
  Gamma-Gamma  conditional on purchasing, what is their average order value?

  CLV = expected purchases over the horizon  x  expected average order value

Both are fitted on a transaction log drawn from the warehouse. Prices are normalized to
a single currency first, because the event stream mixes INR and USD listings and an
unnormalized monetary value would make USD purchases look two orders of magnitude
cheaper than identical INR ones.

Note on the library API: the specification refers to
`lifetimes.utils.summary_data_from_transaction_log`, which does not exist in lifetimes
0.11.3. The real entry point is `summary_data_from_transaction_data`, used below.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure consumer-telemetry root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

from src.config import (
    ARTIFACTS_DIR,
    CLV_BGNBD_PATH,
    CLV_GAMMAGAMMA_PATH,
    DEFAULT_DB_PATH,
    PROCESSED_DATA_DIR,
    TEST_REFERENCE_DATE,
    USD_TO_INR_RATE,
    USER_CLV_PATH,
    rebase_path,
)
from src.features.database import DatabaseManager

CLV_DISTRIBUTION_PLOT_PATH = ARTIFACTS_DIR / "clv_distribution.png"
CLV_SUMMARY_PATH = ARTIFACTS_DIR / "clv_summary.json"
# Simulator ground truth (archetype per user). A warehouse simulated before the raw event
# loader kept the archetype in simulated_events.parquet, which is read when no truth file
# exists yet.
SIMULATED_EVENTS_PATH = PROCESSED_DATA_DIR / "simulated_events.parquet"

DEFAULT_HORIZON_DAYS = 30

# A light L2 penalty. The likelihood will not converge at all without one, but a heavier
# penalty drives the BG/NBD dropout parameter toward zero, which collapses P(alive) to
# 1.0 for every customer and makes the expected-purchase formula divide by zero.
DEFAULT_PENALIZER = 0.001


def load_purchase_transactions(
    con: Optional[duckdb.DuckDBPyConnection] = None,
    db_path: Optional[Union[str, Path]] = None,
    usd_to_inr_rate: float = USD_TO_INR_RATE,
    user_ids: Optional[List[str]] = None,
    reference_date: Optional[Union[str, datetime]] = None,
) -> pl.DataFrame:
    """
    Returns the purchase transaction log with revenue normalized to INR.

    Only `buy` events count as transactions; views and cart adds carry no revenue.
    `user_ids` restricts the log to those users and `reference_date` drops purchases
    after it. The serving API uses both to value one user as of the current moment; the
    batch pipeline passes neither and reads the full log.
    """
    should_close = False
    if con is None:
        db_mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
        con = db_mgr.get_connection()
        should_close = True

    filters = []
    params: Dict[str, Any] = {}
    if user_ids is not None:
        filters.append("AND user_id IN (SELECT UNNEST($user_ids))")
        params["user_ids"] = [str(u) for u in user_ids]
    if reference_date is not None:
        filters.append("AND timestamp <= CAST($reference_date AS TIMESTAMP)")
        params["reference_date"] = (
            reference_date.strftime("%Y-%m-%d %H:%M:%S.%f")
            if isinstance(reference_date, datetime) else str(reference_date)
        )
    filter_sql = "\n              ".join(filters)

    try:
        query = f"""
            SELECT
                user_id,
                CAST(timestamp AS TIMESTAMP) AS timestamp,
                CAST(
                    CASE
                        WHEN currency = 'USD' THEN price * {usd_to_inr_rate}
                        ELSE price
                    END AS DOUBLE
                ) AS revenue_inr
            FROM fact_events
            WHERE action = 'buy'
              AND price IS NOT NULL
              AND price > 0
              {filter_sql}
            ORDER BY user_id, timestamp
        """
        result = con.execute(query, params) if params else con.execute(query)
        return pl.from_arrow(result.to_arrow_table())
    finally:
        if should_close:
            con.close()


def build_rfm_summary(
    transactions: pl.DataFrame,
    observation_period_end: Optional[Union[str, pd.Timestamp]] = None,
    frequency_unit: str = "D",
) -> pd.DataFrame:
    """
    Converts a transaction log into the per-customer summary BG/NBD requires.

    The resulting columns carry the lifetimes conventions:
      frequency       number of *repeat* purchase periods, so a one-off buyer scores 0
      recency         age of the customer at their most recent purchase
      T               total observed age of the customer
      monetary_value  average spend across repeat purchases only
    """
    if transactions.height == 0:
        raise ValueError("Cannot build an RFM summary from an empty transaction log.")

    pdf = transactions.to_pandas()
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])

    period_end = (
        pd.to_datetime(observation_period_end)
        if observation_period_end is not None
        else pdf["timestamp"].max()
    )

    summary = summary_data_from_transaction_data(
        pdf,
        customer_id_col="user_id",
        datetime_col="timestamp",
        monetary_value_col="revenue_inr",
        observation_period_end=period_end,
        freq=frequency_unit,
    )
    return summary


def observed_order_values(transactions: pl.DataFrame) -> pl.DataFrame:
    """
    Returns each user's observed mean order value across all their purchases.

    This differs from the summary table's `monetary_value`, which averages over *repeat*
    purchases only and is therefore 0 for one-off buyers even though they did spend. It
    is the fallback used wherever Gamma-Gamma cannot produce an estimate.
    """
    return transactions.group_by("user_id").agg(
        [
            pl.col("revenue_inr").mean().alias("observed_avg_order_value"),
            pl.len().alias("observed_purchase_count"),
        ]
    )


def fit_bgnbd(summary: pd.DataFrame, penalizer_coef: float = DEFAULT_PENALIZER) -> BetaGeoFitter:
    """Fits the BG/NBD purchase frequency and dropout model."""
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    return bgf


def fit_gamma_gamma(
    summary: pd.DataFrame,
    penalizer_coef: float = DEFAULT_PENALIZER,
) -> Tuple[GammaGammaFitter, pd.DataFrame]:
    """
    Fits the Gamma-Gamma monetary value model on repeat customers only.

    Gamma-Gamma is defined for customers with at least one repeat purchase and a
    positive average order value; one-off and zero buyers carry no within-customer
    spend variance to estimate from. Their monetary value is predicted afterwards from
    the fitted population prior.
    """
    repeat = summary[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)].copy()
    if repeat.empty:
        raise ValueError("No repeat customers available to fit the Gamma-Gamma model.")

    ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
    ggf.fit(repeat["frequency"], repeat["monetary_value"])
    return ggf, repeat


def predict_clv(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    summary: pd.DataFrame,
    transactions: pl.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pl.DataFrame:
    """
    Produces per-user CLV predictions for every customer in the summary.

    CLV is the explicit product the specification asks for: expected purchases over the
    horizon multiplied by expected average order value.

    Two guards matter here, and both exist because of what the fitted parameters turn
    out to be on this data, not as defensive padding:

    1. Gamma-Gamma is only defined for repeat customers. For a one-off buyer the
       conditional expectation collapses to the population prior `p * v / (q - 1)`,
       which is negative whenever the fitted `q` falls below 1, as it does here. Those
       customers fall back to their own observed order value instead.
    2. The expected-purchase formula can return a non-finite value for degenerate
       dropout parameters, so any such row falls back to that user's observed purchase
       rate projected across the horizon.

    The two source columns record which estimate each user actually received, so the
    fallback rate is auditable rather than hidden.
    """
    frame = summary.copy()

    expected_purchases = np.asarray(
        bgf.conditional_expected_number_of_purchases_up_to_time(
            horizon_days, frame["frequency"], frame["recency"], frame["T"]
        ),
        dtype=float,
    )
    probability_alive = np.asarray(
        bgf.conditional_probability_alive(frame["frequency"], frame["recency"], frame["T"]),
        dtype=float,
    )
    gg_order_value = np.asarray(
        ggf.conditional_expected_average_profit(frame["frequency"], frame["monetary_value"]),
        dtype=float,
    )

    observed = observed_order_values(transactions)
    user_ids = frame.index.astype(str).tolist()
    value_map = dict(
        zip(observed["user_id"].to_list(), observed["observed_avg_order_value"].to_list())
    )
    count_map = dict(
        zip(observed["user_id"].to_list(), observed["observed_purchase_count"].to_list())
    )
    observed_value = np.array([float(value_map.get(u, 0.0)) for u in user_ids], dtype=float)
    observed_count = np.array([float(count_map.get(u, 0.0)) for u in user_ids], dtype=float)

    frequency = frame["frequency"].to_numpy(dtype=float)
    observation_days = np.maximum(frame["T"].to_numpy(dtype=float), 1.0)

    gg_is_usable = (frequency > 0) & np.isfinite(gg_order_value) & (gg_order_value > 0)
    order_value = np.where(gg_is_usable, gg_order_value, observed_value)
    order_value_source = np.where(gg_is_usable, "gamma_gamma", "observed")

    naive_purchases = observed_count / observation_days * horizon_days
    purchases_is_usable = np.isfinite(expected_purchases) & (expected_purchases >= 0)
    purchases = np.where(purchases_is_usable, expected_purchases, naive_purchases)
    purchase_source = np.where(purchases_is_usable, "bg_nbd", "observed_rate")

    return (
        pl.DataFrame(
            {
                "user_id": user_ids,
                "frequency": frequency,
                "recency": frame["recency"].to_numpy(dtype=float),
                "T": frame["T"].to_numpy(dtype=float),
                "observed_monetary_value": frame["monetary_value"].to_numpy(dtype=float),
                "observed_avg_order_value": observed_value,
                "observed_purchase_count": observed_count,
                "probability_alive": probability_alive,
                "predicted_purchases": purchases,
                "predicted_avg_order_value": order_value,
                "purchase_estimate_source": purchase_source,
                "order_value_estimate_source": order_value_source,
            }
        )
        .with_columns(
            (pl.col("predicted_purchases") * pl.col("predicted_avg_order_value")).alias(
                "predicted_clv"
            )
        )
        .sort("predicted_clv", descending=True)
    )


def load_user_archetypes(
    events_path: Optional[Union[str, Path]] = None,
) -> Optional[pl.DataFrame]:
    """
    Returns the simulated archetype per user, or None when the event log is absent.

    The archetype is ground truth from the simulator and is not part of the warehouse,
    so it is used only to sanity check the model, never as a feature.
    """
    from src.config import SIMULATION_TRUTH_PATH

    if events_path:
        path = Path(events_path)
    elif SIMULATION_TRUTH_PATH.exists():
        path = SIMULATION_TRUTH_PATH
    else:
        path = SIMULATED_EVENTS_PATH
    if not path.exists() or "archetype" not in pl.read_parquet_schema(path):
        return None

    # Archetypes drift (v1.4), so a user's archetype is the one most of their events carry;
    # ties break by name so the result never depends on row order.
    return (
        pl.read_parquet(path, columns=["user_id", "archetype"])
        .group_by(["user_id", "archetype"]).len()
        .sort(["user_id", "len", "archetype"], descending=[False, True, False])
        .group_by("user_id", maintain_order=True).first()
        .select("user_id", "archetype")
    )


def plot_clv_distribution(
    clv_df: pl.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Saves the CLV distribution histogram alongside the probability-alive spread."""
    out = Path(output_path) if output_path else CLV_DISTRIBUTION_PLOT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    clv = clv_df["predicted_clv"].to_numpy()
    alive = clv_df["probability_alive"].to_numpy()
    purchases = clv_df["predicted_purchases"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].hist(clv, bins=40, color="#2f6f9f", edgecolor="white")
    axes[0].set_xlabel("Predicted CLV (INR)")
    axes[0].set_ylabel("Users")
    axes[0].set_title("CLV Distribution")
    axes[0].grid(axis="y", alpha=0.3, linestyle=":")

    axes[1].hist(purchases, bins=40, color="#4f81a8", edgecolor="white")
    axes[1].set_xlabel("Expected purchases over horizon")
    axes[1].set_ylabel("Users")
    axes[1].set_title("Expected Purchase Frequency")
    axes[1].grid(axis="y", alpha=0.3, linestyle=":")

    axes[2].hist(alive, bins=40, color="#7a9f3f", edgecolor="white")
    axes[2].set_xlabel("P(still active)")
    axes[2].set_ylabel("Users")
    axes[2].set_title("Probability Alive")
    axes[2].grid(axis="y", alpha=0.3, linestyle=":")

    fig.suptitle("Customer Lifetime Value Model Diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def summarize_archetype_clv(clv_df: pl.DataFrame) -> Optional[pl.DataFrame]:
    """Returns mean predicted CLV per simulated archetype, if archetypes are available."""
    archetypes = load_user_archetypes()
    if archetypes is None or "archetype" not in clv_df.columns:
        return None

    return (
        clv_df.group_by("archetype")
        .agg(
            [
                pl.len().alias("users"),
                pl.col("predicted_clv").mean().alias("mean_predicted_clv"),
                pl.col("predicted_purchases").mean().alias("mean_predicted_purchases"),
                pl.col("probability_alive").mean().alias("mean_probability_alive"),
            ]
        )
        .sort("mean_predicted_clv", descending=True)
    )


def run_clv_pipeline(
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    penalizer_coef: float = DEFAULT_PENALIZER,
    output_path: Optional[Union[str, Path]] = None,
    db_path: Optional[Union[str, Path]] = None,
    top_n: int = 10,
    features_dir: Optional[Union[str, Path]] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
    archetype_events_path: Optional[Union[str, Path]] = None,
    reference_date: Optional[str] = TEST_REFERENCE_DATE,
) -> Dict[str, Any]:
    """
    Fits both CLV models, writes predictions, and prints the ranked user readout.

    Only purchases at or before `reference_date` are used, and customer age `T` is
    measured to it, so CLV sees exactly the history the propensity test matrix sees.
    `None` uses the full log up to its last purchase.

    `features_dir` and `artifacts_dir` redirect the prediction table and the model
    artifacts to another dataset's directories. Simulator archetypes are joined only
    for the default synthetic warehouse, or when `archetype_events_path` names an event
    log explicitly: joining the synthetic archetypes onto another warehouse's users
    would silently attach labels that do not belong to them.
    """
    bgnbd_path = rebase_path(CLV_BGNBD_PATH, artifacts_dir)
    gamma_gamma_path = rebase_path(CLV_GAMMAGAMMA_PATH, artifacts_dir)
    summary_out = rebase_path(CLV_SUMMARY_PATH, artifacts_dir)
    print(f"Loading purchase transaction log from the warehouse "
          f"(reference date: {reference_date or 'none, full log'})...")
    transactions = load_purchase_transactions(db_path=db_path, reference_date=reference_date)
    print(f"  {transactions.height} purchase transactions "
          f"across {transactions['user_id'].n_unique()} users")

    summary = build_rfm_summary(transactions, observation_period_end=reference_date)
    repeat_count = int(((summary["frequency"] > 0) & (summary["monetary_value"] > 0)).sum())
    print(f"  Summary built for {len(summary)} customers "
          f"({repeat_count} with repeat purchases)")

    print(f"\nFitting BG/NBD (penalizer {penalizer_coef})...")
    bgf = fit_bgnbd(summary, penalizer_coef=penalizer_coef)
    print(bgf.summary)

    print(f"\nFitting Gamma-Gamma on {repeat_count} repeat customers...")
    ggf, repeat = fit_gamma_gamma(summary, penalizer_coef=penalizer_coef)
    print(ggf.summary)

    # A standard Gamma-Gamma assumption check: within-customer average order value
    # should be roughly independent of purchase frequency. A strong correlation here
    # would invalidate the monetary model.
    correlation = float(repeat[["frequency", "monetary_value"]].corr().iloc[0, 1])
    print(f"\n  Frequency / monetary value correlation: {correlation:.4f} "
          f"(near zero supports the Gamma-Gamma independence assumption)")

    q_value = float(ggf.params_["q"])
    if q_value <= 1.0:
        print(f"\n  Gamma-Gamma q = {q_value:.4f}, which is below 1. The population prior mean")
        print("  p * v / (q - 1) is therefore negative and unusable for one-off buyers. Those")
        print("  customers fall back to their own observed order value; repeat customers keep")
        print("  the fitted Gamma-Gamma estimate.")

    print(f"\nPredicting CLV over a {horizon_days}-day horizon...")
    clv_df = predict_clv(bgf, ggf, summary, transactions, horizon_days=horizon_days)

    fallback_value = int((clv_df["order_value_estimate_source"] == "observed").sum())
    fallback_purchases = int((clv_df["purchase_estimate_source"] == "observed_rate").sum())
    print(f"  Order value from Gamma-Gamma: {clv_df.height - fallback_value}, "
          f"from observed history: {fallback_value}")
    print(f"  Purchases from BG/NBD: {clv_df.height - fallback_purchases}, "
          f"from observed rate: {fallback_purchases}")

    if not clv_df["predicted_clv"].is_finite().all():
        raise ValueError("Non-finite CLV predictions produced")
    if clv_df["predicted_clv"].min() < 0.0:
        raise ValueError("Negative CLV predictions produced")

    archetypes = (
        load_user_archetypes(archetype_events_path)
        if archetype_events_path is not None or db_path is None
        else None
    )
    if archetypes is not None:
        clv_df = clv_df.join(archetypes, on="user_id", how="left")

    out_path = Path(output_path) if output_path else rebase_path(USER_CLV_PATH, features_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clv_df.write_parquet(out_path)
    print(f"  Saved {clv_df.height} CLV predictions to {out_path}")

    display_cols = [
        "user_id",
        "frequency",
        "predicted_purchases",
        "predicted_avg_order_value",
        "predicted_clv",
        "probability_alive",
        "order_value_estimate_source",
    ]
    if "archetype" in clv_df.columns:
        display_cols.append("archetype")

    print(f"\n--- Top {top_n} Users by Predicted CLV ---")
    print(clv_df.select(display_cols).head(top_n))

    print(f"\n--- Bottom {top_n} Users by Predicted CLV ---")
    print(clv_df.select(display_cols).tail(top_n))

    archetype_summary = summarize_archetype_clv(clv_df)
    if archetype_summary is not None:
        print("\n--- Mean Predicted CLV by Simulated Archetype ---")
        print(archetype_summary)
        print("\n  The archetype is simulator ground truth and is never used as a model")
        print("  feature. Its ordering here is an independent check that CLV recovers the")
        print("  latent behavior that generated the data.")

    plot_path = plot_clv_distribution(
        clv_df, output_path=rebase_path(CLV_DISTRIBUTION_PLOT_PATH, artifacts_dir)
    )

    # The lifetimes fitters attach a lambda to the instance during fit, which both pickle
    # and joblib refuse to serialize. The library's own save_model uses dill and can drop
    # that method, so it is used here in place of joblib.
    bgnbd_path.parent.mkdir(parents=True, exist_ok=True)
    bgf.save_model(str(bgnbd_path), save_generate_data_method=False)
    ggf.save_model(str(gamma_gamma_path), save_generate_data_method=False)

    payload: Dict[str, Any] = {
        "reference_date": reference_date,
        "horizon_days": horizon_days,
        "penalizer_coef": penalizer_coef,
        "n_customers": int(clv_df.height),
        "n_repeat_customers": repeat_count,
        "n_transactions": int(transactions.height),
        "frequency_monetary_correlation": correlation,
        "bgnbd_params": {k: float(v) for k, v in bgf.params_.items()},
        "gamma_gamma_params": {k: float(v) for k, v in ggf.params_.items()},
        "gamma_gamma_q_below_one": q_value <= 1.0,
        # The Gamma-Gamma population mean order value p * v / (q - 1) only exists for q > 1.
        "gamma_gamma_mean_undefined": bool(q_value <= 1.0),
        "order_value_fallback_count": fallback_value,
        "purchase_fallback_count": fallback_purchases,
        "total_predicted_clv": float(clv_df["predicted_clv"].sum()),
        "mean_predicted_clv": float(clv_df["predicted_clv"].mean()),
        "median_predicted_clv": float(clv_df["predicted_clv"].median()),
        "top_users": clv_df.select(display_cols).head(top_n).to_dicts(),
        "bottom_users": clv_df.select(display_cols).tail(top_n).to_dicts(),
    }
    if archetype_summary is not None:
        payload["archetype_summary"] = archetype_summary.to_dicts()

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved BG/NBD model to      {bgnbd_path}")
    print(f"Saved Gamma-Gamma model to {gamma_gamma_path}")
    print(f"Saved diagnostics plot to  {plot_path}")
    print(f"Saved summary to           {summary_out}")

    return {
        **payload,
        "bgnbd": bgf,
        "gamma_gamma": ggf,
        "summary_table": summary,
        "clv": clv_df,
        "archetype_summary": archetype_summary,
        "paths": {
            "predictions": out_path,
            "bgnbd": bgnbd_path,
            "gamma_gamma": gamma_gamma_path,
            "plot": plot_path,
            "summary": summary_out,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit BG/NBD and Gamma-Gamma CLV models.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_DAYS, help="Horizon in days")
    parser.add_argument("--penalizer", type=float, default=DEFAULT_PENALIZER, help="L2 penalizer")
    parser.add_argument("--top-n", type=int, default=10, help="Users listed at each extreme")
    parser.add_argument("--db-path", type=str, default=None, help="Warehouse to read (default: synthetic)")
    parser.add_argument("--features-dir", type=str, default=None, help="Directory for user_clv.parquet")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Directory for model outputs")
    parser.add_argument("--reference-date", type=str, default=TEST_REFERENCE_DATE,
                        help="Use purchases at or before this timestamp (default: test reference date)")
    args = parser.parse_args()

    run_clv_pipeline(
        horizon_days=args.horizon,
        penalizer_coef=args.penalizer,
        top_n=args.top_n,
        db_path=args.db_path,
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
        reference_date=args.reference_date,
    )

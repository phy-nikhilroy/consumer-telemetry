"""
Out-of-time validation for customer lifetime value.

`clv.py` fits BG/NBD and Gamma-Gamma and predicts forward, but nothing there checks the
predictions against what customers then did. This module does, in two ways:

* `evaluate_clv_holdout` fits both models on purchases up to a calibration end, predicts
  the following holdout window, and compares predictions with the purchases and spend
  that actually happened, next to a naive purchase-rate baseline and three order-value
  estimators (Gamma-Gamma, the user's observed mean, and a shrunk mean).
* `evaluate_clv_regressor` trains a Tweedie LightGBM regressor on the feature store to
  predict spend over the same horizon, and reports it beside BG/NBD on the users that
  BG/NBD can score.

Every input is cut at its reference date: nothing after the calibration end reaches a
fit, and nothing after the holdout end reaches an actual.

Run directly to write `reports/v1_1_clv.json`. Nothing under `data/` or the model
artifacts directory is written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from src.config import (
    CLV_HOLDOUT_CALIBRATION_END,
    CLV_HOLDOUT_DAYS,
    CLV_ORDER_VALUE_SHRINKAGE_K,
    DEFAULT_DB_PATH,
    RANDOM_SEED,
    USD_TO_INR_RATE,
)
from src.features.database import DatabaseManager
from src.features.feature_store import (
    NON_FEATURE_COLUMNS,
    build_feature_matrix,
    fill_missing_features,
)
from src.models.clv import (
    DEFAULT_PENALIZER,
    build_rfm_summary,
    fit_bgnbd,
    fit_gamma_gamma,
    load_purchase_transactions,
    predict_clv,
)

# A user with no buy event in a window made zero purchases and spent zero: the left join
# leaves them null only because the window query returns buyers alone. This is a true
# count, not a missing feature, so it is the one place a null legitimately becomes zero.
NO_PURCHASES_IN_WINDOW = 0.0

FREQUENCY_BUCKETS = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6+", 6, None))
REGRESSOR_PARAMS: Dict[str, Any] = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "random_state": RANDOM_SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def _ts(value: Union[str, datetime]) -> datetime:
    """Parses a timestamp string, or passes a datetime through."""
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value).strip())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation; 0.0 when either side is constant and the rank is undefined."""
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    rho = float(spearmanr(a, b).statistic)
    return rho if np.isfinite(rho) else 0.0


def shrunk_order_value(
    n_orders: np.ndarray,
    observed_mean: np.ndarray,
    global_mean: float,
    k: float = CLV_ORDER_VALUE_SHRINKAGE_K,
) -> np.ndarray:
    """
    Returns `(n_i * mean_i + K * global_mean) / (n_i + K)` per user.

    The user's own mean counts with the weight of their order count and the global mean
    with the weight of K pseudo-orders, so a one-order user sits close to the global mean
    and a frequent buyer close to their own.
    """
    n = np.asarray(n_orders, dtype=float)
    mean = np.asarray(observed_mean, dtype=float)
    return (n * mean + k * float(global_mean)) / (n + k)


def _window_purchases(
    db_path: Union[str, Path],
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """Per-user purchase count and INR spend for buys in `(start, end]`."""
    con = DatabaseManager(db_path=db_path).get_connection()
    try:
        table = con.execute(
            f"""
            SELECT
                user_id,
                COUNT(*) AS actual_purchases,
                SUM(CASE WHEN currency = 'USD' THEN price * {USD_TO_INR_RATE}
                         ELSE price END) AS actual_spend
            FROM fact_events
            WHERE action = 'buy'
              AND price IS NOT NULL AND price > 0
              AND timestamp > CAST($start AS TIMESTAMP)
              AND timestamp <= CAST($end AS TIMESTAMP)
            GROUP BY user_id
            """,
            {"start": start.strftime("%Y-%m-%d %H:%M:%S.%f"),
             "end": end.strftime("%Y-%m-%d %H:%M:%S.%f")},
        ).to_arrow_table()
    finally:
        con.close()
    frame = pl.from_arrow(table)
    return frame.with_columns(
        pl.col("user_id").cast(pl.Utf8),
        pl.col("actual_purchases").cast(pl.Float64),
        pl.col("actual_spend").cast(pl.Float64),
    )


def _holdout_frame(
    db_path: Union[str, Path],
    calibration_end: str,
    holdout_days: int,
) -> Dict[str, Any]:
    """
    Fits BG/NBD and Gamma-Gamma on purchases up to `calibration_end` and joins each
    calibration customer's predictions with their actual holdout behaviour.
    """
    cutoff = _ts(calibration_end)
    holdout_end = cutoff + timedelta(days=holdout_days)

    transactions = load_purchase_transactions(db_path=db_path, reference_date=cutoff)
    if transactions.height == 0:
        raise ValueError(f"no purchases at or before {calibration_end}")
    summary = build_rfm_summary(transactions, observation_period_end=cutoff)
    bgf = fit_bgnbd(summary, penalizer_coef=DEFAULT_PENALIZER)
    ggf, _ = fit_gamma_gamma(summary, penalizer_coef=DEFAULT_PENALIZER)
    predictions = predict_clv(bgf, ggf, summary, transactions, horizon_days=holdout_days)

    actual = _window_purchases(db_path, cutoff, holdout_end)
    global_mean = float(transactions["revenue_inr"].mean())
    frame = (
        predictions.join(actual, on="user_id", how="left")
        .with_columns(
            pl.col("actual_purchases").fill_null(NO_PURCHASES_IN_WINDOW),
            pl.col("actual_spend").fill_null(NO_PURCHASES_IN_WINDOW),
            # Purchases per observed day before the cutoff, projected over the holdout.
            (pl.col("observed_purchase_count") / pl.col("T").clip(lower_bound=1.0)
             * float(holdout_days)).alias("naive_purchases"),
        )
        .with_columns(
            (pl.col("predicted_purchases") * pl.col("predicted_avg_order_value"))
            .alias("predicted_spend"),
            (pl.col("naive_purchases") * pl.col("observed_avg_order_value"))
            .alias("naive_spend"),
        )
        .sort("user_id")
    )
    frame = frame.with_columns(
        pl.Series(
            "shrunk_order_value",
            shrunk_order_value(
                frame["observed_purchase_count"].to_numpy(),
                frame["observed_avg_order_value"].to_numpy(),
                global_mean,
            ),
        )
    )
    return {"frame": frame, "bgnbd": bgf, "gamma_gamma": ggf, "global_mean": global_mean}


def evaluate_clv_holdout(
    db_path: Union[str, Path],
    calibration_end: str,
    holdout_days: int,
) -> Dict[str, Any]:
    """
    Scores BG/NBD and Gamma-Gamma against what calibration customers did next.

    Customers are users with at least one purchase at or before `calibration_end`.
    Actuals are their purchases in `(calibration_end, calibration_end + holdout_days]`.
    """
    fitted = _holdout_frame(db_path, calibration_end, holdout_days)
    frame: pl.DataFrame = fitted["frame"]

    predicted = frame["predicted_purchases"].to_numpy()
    naive = frame["naive_purchases"].to_numpy()
    actual = frame["actual_purchases"].to_numpy()
    actual_spend = frame["actual_spend"].to_numpy()

    buckets: Dict[str, Dict[str, float]] = {}
    frequency = frame["frequency"].to_numpy()
    for label, lo, hi in FREQUENCY_BUCKETS:
        mask = frequency >= lo if hi is None else (frequency >= lo) & (frequency <= hi)
        n = int(mask.sum())
        buckets[label] = {
            "n": n,
            "mean_predicted_purchases": float(predicted[mask].mean()) if n else 0.0,
            "mean_actual_purchases": float(actual[mask].mean()) if n else 0.0,
        }

    # Order value is judged only where the holdout has orders to average.
    bought = frame.filter(pl.col("actual_purchases") > 0)
    holdout_aov = (bought["actual_spend"] / bought["actual_purchases"]).to_numpy()

    def aov_mae(column: str) -> float:
        if bought.height == 0:
            return 0.0
        return float(np.abs(bought[column].to_numpy() - holdout_aov).mean())

    q_value = float(fitted["gamma_gamma"].params_["q"])
    return {
        "calibration_end": str(calibration_end),
        "holdout_days": int(holdout_days),
        "n_customers": int(frame.height),
        "n_customers_with_holdout_purchase": int(bought.height),
        "actual_total_purchases": int(actual.sum()),
        "predicted_total_purchases": float(predicted.sum()),
        "naive_total_purchases": float(naive.sum()),
        "mae_purchases_bgnbd": float(np.abs(predicted - actual).mean()),
        "mae_purchases_naive": float(np.abs(naive - actual).mean()),
        "spearman_spend_bgnbd_gg": _spearman(frame["predicted_spend"].to_numpy(), actual_spend),
        "spearman_spend_naive": _spearman(frame["naive_spend"].to_numpy(), actual_spend),
        "by_frequency_bucket": buckets,
        "order_value_mae": {
            # Gamma-Gamma where it is defined (repeat buyers), observed mean otherwise,
            # exactly as predict_clv serves it.
            "gamma_gamma": aov_mae("predicted_avg_order_value"),
            "observed_mean": aov_mae("observed_avg_order_value"),
            "shrunk": aov_mae("shrunk_order_value"),
        },
        "shrinkage_k": CLV_ORDER_VALUE_SHRINKAGE_K,
        "global_mean_order_value": fitted["global_mean"],
        "bgnbd_params": {k: float(v) for k, v in fitted["bgnbd"].params_.items()},
        "gamma_gamma_params": {k: float(v) for k, v in fitted["gamma_gamma"].params_.items()},
        "gamma_gamma_mean_undefined": bool(q_value <= 1.0),
    }


def _regression_matrix(
    db_path: Union[str, Path],
    reference_date: str,
    horizon_days: int,
) -> pl.DataFrame:
    """Feature matrix at `reference_date` joined with INR spend over the next horizon."""
    ref = _ts(reference_date)
    features = build_feature_matrix(
        reference_date=ref.strftime("%Y-%m-%d %H:%M:%S"),
        prediction_window_days=horizon_days,
        db_path=db_path,
    )
    spend = _window_purchases(db_path, ref, ref + timedelta(days=horizon_days))
    return (
        features.join(spend.select(["user_id", "actual_spend"]), on="user_id", how="left")
        .with_columns(pl.col("actual_spend").fill_null(NO_PURCHASES_IN_WINDOW))
        .sort("user_id")
    )


def evaluate_clv_regressor(
    db_path: Union[str, Path],
    train_ref: str = "2026-10-09 00:00:00",
    test_ref: str = "2026-10-18 00:00:00",
    horizon_days: int = 9,
) -> Dict[str, Any]:
    """
    Compares a Tweedie LightGBM spend regressor with BG/NBD and Gamma-Gamma.

    The regressor trains on features at `train_ref` against spend in
    `(train_ref, train_ref + horizon]` and is tested on features at `test_ref` against
    spend in `(test_ref, test_ref + horizon]`. BG/NBD is fitted on purchases up to
    `test_ref`, so it scores only users who had bought by then; both models are also
    reported on that subset.
    """
    if _ts(train_ref) + timedelta(days=horizon_days) > _ts(test_ref):
        raise ValueError("train target window overlaps the test reference date")

    train = _regression_matrix(db_path, train_ref, horizon_days)
    test = _regression_matrix(db_path, test_ref, horizon_days)
    excluded = set(NON_FEATURE_COLUMNS) | {"actual_spend"}
    feature_names = [
        c for c, dtype in train.schema.items() if c not in excluded and dtype.is_numeric()
    ]
    X_train = fill_missing_features(train, feature_names).to_numpy()
    X_test = fill_missing_features(test, feature_names).to_numpy()
    y_train = train["actual_spend"].to_numpy()
    y_test = test["actual_spend"].to_numpy()

    model = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
    model.fit(X_train, y_train)
    regressor_pred = np.clip(model.predict(X_test), 0.0, None)

    bgnbd = _holdout_frame(db_path, test_ref, horizon_days)["frame"].select(
        ["user_id", "predicted_spend"]
    )
    scored = test.select("user_id").with_columns(
        pl.Series("regressor_pred", regressor_pred), pl.Series("actual", y_test)
    ).join(bgnbd, on="user_id", how="inner").sort("user_id")

    def block(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
        return {
            "spearman": _spearman(pred, truth),
            "mae": float(np.abs(pred - truth).mean()) if len(truth) else 0.0,
        }

    return {
        "train_ref": str(train_ref),
        "test_ref": str(test_ref),
        "horizon_days": int(horizon_days),
        "n_train_users": int(train.height),
        "n_test_users": int(test.height),
        "n_features": len(feature_names),
        "regressor_all_test_users": block(regressor_pred, y_test),
        "n_bgnbd_scorable_users": int(scored.height),
        "regressor_on_bgnbd_subset": block(
            scored["regressor_pred"].to_numpy(), scored["actual"].to_numpy()
        ),
        "bgnbd_gg_on_bgnbd_subset": block(
            scored["predicted_spend"].to_numpy(), scored["actual"].to_numpy()
        ),
    }


def run_clv_validation(db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Runs the v1.1 holdout and regressor comparison and writes `reports/v1_1_clv.json`."""
    from src.release.reporting import write_report

    db = db_path or DEFAULT_DB_PATH
    results = evaluate_clv_holdout(db, CLV_HOLDOUT_CALIBRATION_END, CLV_HOLDOUT_DAYS)
    results["regressor_comparison"] = evaluate_clv_regressor(db)
    path = write_report("v1_1_clv", "1.1", results)
    print(f"Holdout customers: {results['n_customers']}, "
          f"actual purchases {results['actual_total_purchases']}, "
          f"predicted {results['predicted_total_purchases']:.1f}")
    print(f"Purchase MAE  BG/NBD {results['mae_purchases_bgnbd']:.3f}  "
          f"naive {results['mae_purchases_naive']:.3f}")
    print(f"Spend Spearman  BG/NBD+GG {results['spearman_spend_bgnbd_gg']:.3f}  "
          f"naive {results['spearman_spend_naive']:.3f}")
    print(f"Order value MAE {results['order_value_mae']}")
    print(f"Saved report to {path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate CLV on a holdout window.")
    parser.add_argument("--db-path", type=str, default=None, help="Warehouse to read")
    args = parser.parse_args()
    run_clv_validation(db_path=args.db_path)

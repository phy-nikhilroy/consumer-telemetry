"""Recover a planted price effect: simulate with price_beta in {0, 4, 8}, then measure it from the warehouse.

Two measurements per price_beta, using only warehouse tables (no truth columns):
- the population mean of the feature `chose_cheaper_app_rate` over users where it is not null;
- a logistic regression on informed purchases of "bought on the app of the later view" against the
  log price ratio of the two apps and an indicator that the later app is the user's dominant app,
  with a user-clustered bootstrap CI on the log price ratio coefficient.

Writes `reports/v1_4_price_effect.json`. Simulations live under data/experiments/v1_4_price.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CROSS_APP_PRICE_LOOKBACK_HOURS, PROJECT_ROOT, RANDOM_SEED

DEFAULT_OUT = PROJECT_ROOT / "data" / "experiments" / "v1_4_price"
PRICE_BETAS = (0.0, 4.0, 8.0)
SEEDS = (RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2)
USERS = 500
DAYS = 45
LOAD_WORKERS = 4
FEATURE_NAMES = ("log_price_ratio", "later_is_dominant")


def informed_purchase_frame(db_path: Path, lookback_hours: int = CROSS_APP_PRICE_LOOKBACK_HOURS) -> pl.DataFrame:
    """One row per informed buy whose last two distinct-app observations explain the choice.

    For each buy, the user's earlier observations of the same product on any app within
    `lookback_hours` (excluding the buy screen itself) give the later view (latest observation)
    and the earlier view (latest observation on a different app). Rows keep buys made on one
    of those two apps. Columns: user_id, y (bought on the later-view app), log_price_ratio
    (log of later price over earlier price, INR), later_is_dominant (later app is the user's
    most frequent app in fact_events).
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = pl.from_arrow(con.execute(f"""
            WITH buys AS (
                SELECT event_id, user_id, timestamp, app_name, canonical_product_id
                FROM fact_events WHERE action = 'buy' AND canonical_product_id IS NOT NULL
            ),
            prior AS (
                SELECT b.event_id AS buy_id, b.user_id, b.app_name AS buy_app,
                       o.app_name, o.timestamp AS ts, o.price_inr,
                       ROW_NUMBER() OVER (PARTITION BY b.event_id, o.app_name
                                          ORDER BY o.timestamp DESC, o.observation_id DESC) AS rn
                FROM buys b
                JOIN fact_price_observations o
                  ON o.user_id = b.user_id AND o.canonical_product_id = b.canonical_product_id
                 AND o.source_event_id <> b.event_id
                 AND o.timestamp <= b.timestamp
                 AND o.timestamp >= b.timestamp - INTERVAL ({int(lookback_hours)}) HOUR
                WHERE o.price_inr IS NOT NULL AND o.price_inr > 0
            ),
            per_app AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY buy_id ORDER BY ts DESC, app_name) AS app_rank
                FROM prior WHERE rn = 1
            ),
            dom AS (
                SELECT user_id, app_name AS dominant_app FROM (
                    SELECT user_id, app_name,
                           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC, app_name) AS r
                    FROM fact_events GROUP BY user_id, app_name) WHERE r = 1
            )
            SELECT l.user_id, l.buy_app, l.app_name AS later_app, e.app_name AS earlier_app,
                   l.price_inr AS later_price, e.price_inr AS earlier_price, d.dominant_app
            FROM per_app l
            JOIN per_app e ON e.buy_id = l.buy_id AND e.app_rank = 2
            JOIN dom d ON d.user_id = l.user_id
            WHERE l.app_rank = 1
        """).to_arrow_table())
    finally:
        con.close()
    df = df.filter((pl.col("buy_app") == pl.col("later_app")) | (pl.col("buy_app") == pl.col("earlier_app")))
    return df.select(
        "user_id",
        (pl.col("buy_app") == pl.col("later_app")).cast(pl.Int8).alias("y"),
        (pl.col("later_price") / pl.col("earlier_price")).log().alias("log_price_ratio"),
        (pl.col("later_app") == pl.col("dominant_app")).cast(pl.Int8).alias("later_is_dominant"),
    )


def _fit_coef(x: np.ndarray, y: np.ndarray) -> float:
    """Unpenalised logistic coefficient on column 0 of x, or NaN when y has one class."""
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=np.inf, max_iter=1000)
    model.fit(x, y)
    return float(model.coef_[0][0])


def price_ratio_regression(frame: pl.DataFrame, n_boot: int = 500, seed: int = RANDOM_SEED,
                           alpha: float = 0.05) -> Dict[str, Any]:
    """Logistic fit of y on log_price_ratio and later_is_dominant, with a user-clustered bootstrap CI.

    Returns n, coefficients, and `ci` (percentile interval of the log_price_ratio coefficient;
    NaN values when the data cannot support a fit).
    """
    nan = float("nan")
    out: Dict[str, Any] = {"n": frame.height, "coef_log_price_ratio": nan, "coef_later_is_dominant": nan,
                           "ci": {"lo": nan, "hi": nan, "n_boot_ok": 0}}
    if frame.height == 0:
        return out
    x = frame.select(*FEATURE_NAMES).to_numpy().astype(float)
    y = frame["y"].to_numpy().astype(int)
    if len(np.unique(y)) < 2:
        return out
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=np.inf, max_iter=1000).fit(x, y)
    out["coef_log_price_ratio"] = float(model.coef_[0][0])
    out["coef_later_is_dominant"] = float(model.coef_[0][1])
    users = frame["user_id"].to_numpy()
    uniq, inverse = np.unique(users, return_inverse=True)
    rows_by_user = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    coefs: List[float] = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([rows_by_user[i] for i in pick])
        c = _fit_coef(x[idx], y[idx])
        if not np.isnan(c):
            coefs.append(c)
    if coefs:
        lo, hi = np.percentile(coefs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out["ci"] = {"lo": float(lo), "hi": float(hi), "n_boot_ok": len(coefs)}
    return out


def mean_chose_cheaper(db_path: Path) -> Dict[str, Any]:
    """Population mean of `chose_cheaper_app_rate` over users where it is not null."""
    from src.features.behavioral_features import compute_behavioral_features

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        last = con.execute("SELECT MAX(timestamp) FROM fact_events").fetchone()[0]
        ref = last + timedelta(seconds=1)
        feats = compute_behavioral_features(ref, con=con)
    finally:
        con.close()
    vals = feats["chose_cheaper_app_rate"].drop_nulls()
    return {"users_with_feature": int(vals.len()), "users": feats.height,
            "mean": float(vals.mean()) if vals.len() else float("nan")}


def run_one(beta: float, seed: int, out_dir: Path, users: int, days: int) -> Dict[str, Any]:
    """Simulates one (beta, seed) cell under `out_dir`, returning its feature mean and regression frame."""
    from src.features.sessionizer import run_sessionization_pipeline
    from src.ingestion.simulate_users import run_simulation

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "warehouse.duckdb"
    run_simulation(num_users=users, days=days, db_path=str(db_path), output_parquet=str(out_dir / "events.parquet"),
                   seed=seed, workers=LOAD_WORKERS, truth_parquet=str(out_dir / "truth.parquet"),
                   price_beta=beta)
    run_sessionization_pipeline(db_path=db_path)
    return {"feature": mean_chose_cheaper(db_path), "frame": informed_purchase_frame(db_path)}


def run_price_effect(out_root: Path = DEFAULT_OUT, betas: Sequence[float] = PRICE_BETAS,
                     seeds: Sequence[int] = SEEDS, users: int = USERS, days: int = DAYS,
                     n_boot: int = 500) -> Dict[str, Any]:
    """Runs every (beta, seed) cell and reports the dose response and per-beta regression."""
    levels: Dict[str, Any] = {}
    for beta in betas:
        cells, frames = [], []
        for seed in seeds:
            cell = run_one(beta, seed, Path(out_root) / f"beta_{beta:g}" / f"seed_{seed}", users, days)
            cells.append({"seed": seed, **cell["feature"], "informed_rows": cell["frame"].height})
            frames.append(cell["frame"].with_columns(
                (pl.lit(f"{seed}_") + pl.col("user_id")).alias("user_id")))
        pooled = pl.concat(frames) if frames else pl.DataFrame()
        means = [c["mean"] for c in cells if c["mean"] == c["mean"]]
        levels[f"{beta:g}"] = {
            "price_beta": float(beta),
            "chose_cheaper_mean": float(np.mean(means)) if means else float("nan"),
            "chose_cheaper_by_seed": cells,
            "regression": price_ratio_regression(pooled, n_boot=n_boot),
        }
    return {"users": users, "days": days, "seeds": list(seeds), "price_betas": [float(b) for b in betas],
            "levels": levels}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recover a planted price effect from simulated warehouses.")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Root for isolated simulation outputs")
    parser.add_argument("--betas", type=float, nargs="+", default=list(PRICE_BETAS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--users", type=int, default=USERS)
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    from src.release.reporting import write_report

    results = run_price_effect(Path(args.out), args.betas, args.seeds, args.users, args.days, args.n_boot)
    print(f"Wrote {write_report('v1_4_price_effect', '1.4', results, seed=RANDOM_SEED)}")

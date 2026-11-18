"""
Cross-App and Behavioral Feature Store Engineering.
Computes multi-app entropy, engagement funnels, session dynamics, and price sensitivity features.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

# Ensure consumer-telemetry root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import polars as pl
from src.config import (
    BEHAVIORAL_FEATURES_PATH,
    CROSS_APP_PRICE_LOOKBACK_HOURS,
    DEFAULT_DB_PATH,
    USD_TO_INR_RATE,
)
from src.features.database import DatabaseManager

DEFAULT_BEHAVIORAL_PATH = BEHAVIORAL_FEATURES_PATH


def compute_behavioral_features(
    reference_date: Union[str, datetime],
    con: Optional[duckdb.DuckDBPyConnection] = None,
    db_path: Optional[Union[str, Path]] = None,
    usd_to_inr_rate: float = USD_TO_INR_RATE,
    user_ids: Optional[Sequence[str]] = None,
    lookback_hours: int = CROSS_APP_PRICE_LOOKBACK_HOURS,
) -> pl.DataFrame:
    """
    Computes 15+ behavioral and cross-app features per user.
    Every table read is bounded by reference_date: events and price observations by
    timestamp, comparisons by both first_viewed and last_viewed, to prevent target leakage.

    Price features read only stored per-event values (price_inr, discount_pct) and
    per-user price observations, never a product-level average:
    - avg_discount_pct_on_purchases, pct_purchases_with_discount: over buy events, from the
      discount shown on the screen (a missing discount counts as none); 0.0 with no buys.
    - informed_purchase_count: buys for which the user saw the same product priced on a
      different app within [buy - lookback_hours, buy].
    - chose_cheaper_app_rate: share of informed buys whose price_inr is at most the lowest
      such other-app price; null with no informed buy.
    - avg_abs_price_gap_pct: mean absolute price gap across the user's comparisons, as a
      percent of the first price; null with none.

    `user_ids` restricts the computation to those users, for per-user scoring at serving
    time. Every feature is a per-user aggregate, so the restricted result equals the
    matching rows of the unrestricted one.
    """
    if isinstance(reference_date, datetime):
        ref_str = reference_date.strftime("%Y-%m-%d %H:%M:%S")
    else:
        ref_str = str(reference_date).strip()

    should_close = False
    if con is None:
        db_mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
        con = db_mgr.get_connection()
        should_close = True

    event_filter = "AND fe.user_id IN (SELECT UNNEST($user_ids))" if user_ids is not None else ""
    comparison_filter = "AND user_id IN (SELECT UNNEST($user_ids))" if user_ids is not None else ""
    params = {"user_ids": [str(u) for u in user_ids]} if user_ids is not None else None

    try:
        query = f"""
            WITH eligible_events AS (
                SELECT
                    fe.event_id,
                    fe.user_id,
                    fe.timestamp,
                    fe.session_id,
                    fe.app_name,
                    fe.canonical_product_id,
                    fe.action,
                    fe.price,
                    fe.currency,
                    fe.price_inr,
                    fe.discount_pct,
                    CASE
                        WHEN fe.currency = 'USD' THEN fe.price * {usd_to_inr_rate}
                        ELSE fe.price
                    END AS norm_price,
                    dp.category_l1
                FROM fact_events fe
                LEFT JOIN dim_products dp ON fe.canonical_product_id = dp.canonical_product_id
                WHERE fe.timestamp <= CAST('{ref_str}' AS TIMESTAMP)
                {event_filter}
            ),
            app_counts AS (
                SELECT user_id, app_name, COUNT(*) AS app_events
                FROM eligible_events
                GROUP BY user_id, app_name
            ),
            user_app_totals AS (
                SELECT
                    user_id,
                    SUM(app_events) AS total_events,
                    MAX(app_events) AS max_app_events,
                    COUNT(DISTINCT app_name) AS num_apps_used
                FROM app_counts
                GROUP BY user_id
            ),
            entropy_calc AS (
                SELECT
                    ac.user_id,
                    ROUND(-SUM((CAST(ac.app_events AS FLOAT) / ut.total_events) * LOG2(CAST(ac.app_events AS FLOAT) / ut.total_events)), 4) AS app_entropy,
                    ROUND(CAST(ut.max_app_events AS FLOAT) / ut.total_events, 4) AS dominant_app_share
                FROM app_counts ac
                JOIN user_app_totals ut ON ac.user_id = ut.user_id
                GROUP BY ac.user_id, ut.total_events, ut.max_app_events
            ),
            session_durations AS (
                SELECT
                    user_id,
                    session_id,
                    MIN(timestamp) AS session_start,
                    EPOCH(MAX(timestamp) - MIN(timestamp)) / 60.0 AS duration_minutes
                FROM eligible_events
                WHERE session_id IS NOT NULL
                GROUP BY user_id, session_id
            ),
            session_stats AS (
                SELECT
                    user_id,
                    ROUND(COALESCE(AVG(duration_minutes), 0.0), 2) AS avg_session_duration_minutes
                FROM session_durations
                GROUP BY user_id
            ),
            session_gaps AS (
                SELECT
                    user_id,
                    EPOCH(session_start - LAG(session_start) OVER (PARTITION BY user_id ORDER BY session_start)) / 3600.0 AS gap_hours
                FROM session_durations
            ),
            session_gap_stats AS (
                SELECT
                    user_id,
                    ROUND(COALESCE(AVG(gap_hours), 0.0), 2) AS avg_time_between_sessions_hours
                FROM session_gaps
                GROUP BY user_id
            ),
            cross_app_counts AS (
                SELECT
                    user_id,
                    COUNT(*) AS cross_app_comparison_count,
                    ROUND(AVG(
                        ABS(CAST(price_last_inr AS DOUBLE) - CAST(price_first_inr AS DOUBLE))
                        / NULLIF(CAST(price_first_inr AS DOUBLE), 0.0) * 100.0
                    ), 2) AS avg_abs_price_gap_pct
                FROM fact_cross_app_comparisons
                WHERE first_viewed <= CAST('{ref_str}' AS TIMESTAMP)
                  AND last_viewed <= CAST('{ref_str}' AS TIMESTAMP)
                {comparison_filter}
                GROUP BY user_id
            ),
            informed_purchases AS (
                SELECT
                    b.user_id,
                    b.event_id,
                    CASE WHEN b.price_inr <= MIN(o.price_inr) THEN 1.0 ELSE 0.0 END AS chose_cheaper
                FROM eligible_events b
                JOIN fact_price_observations o
                  ON o.user_id = b.user_id
                 AND o.canonical_product_id = b.canonical_product_id
                 AND o.app_name <> b.app_name
                 AND o.timestamp <= b.timestamp
                 AND o.timestamp >= b.timestamp - INTERVAL ({int(lookback_hours)}) HOUR
                 AND o.timestamp <= CAST('{ref_str}' AS TIMESTAMP)
                WHERE b.action = 'buy'
                GROUP BY b.user_id, b.event_id, b.price_inr
            ),
            informed_stats AS (
                SELECT
                    user_id,
                    COUNT(*) AS informed_purchase_count,
                    ROUND(AVG(chose_cheaper), 4) AS chose_cheaper_app_rate
                FROM informed_purchases
                GROUP BY user_id
            ),
            user_features AS (
                SELECT
                    ee.user_id,
                    -- CROSS-APP DYNAMICS
                    uat.num_apps_used,
                    ec.app_entropy,
                    ec.dominant_app_share,
                    COALESCE(cac.cross_app_comparison_count, 0) AS cross_app_comparison_count,
                    cac.avg_abs_price_gap_pct,
                    COALESCE(ist.informed_purchase_count, 0) AS informed_purchase_count,
                    ist.chose_cheaper_app_rate,

                    -- ENGAGEMENT PATTERNS
                    ROUND(COALESCE(
                        CAST(COUNT(*) FILTER (WHERE ee.action = 'cart') AS FLOAT) /
                        NULLIF(COUNT(*) FILTER (WHERE ee.action = 'view'), 0),
                        0.0
                    ), 4) AS view_to_cart_ratio,

                    ROUND(COALESCE(
                        CAST(COUNT(*) FILTER (WHERE ee.action = 'buy') AS FLOAT) /
                        NULLIF(COUNT(*) FILTER (WHERE ee.action = 'cart'), 0),
                        0.0
                    ), 4) AS cart_to_purchase_ratio,

                    COALESCE(ss.avg_session_duration_minutes, 0.0) AS avg_session_duration_minutes,
                    COALESCE(sgs.avg_time_between_sessions_hours, 0.0) AS avg_time_between_sessions_hours,

                    -- PRICE SENSITIVITY
                    ROUND(COALESCE(
                        AVG(CAST(COALESCE(ee.discount_pct, 0) AS DOUBLE)) FILTER (WHERE ee.action = 'buy'),
                        0.0
                    ), 2) AS avg_discount_pct_on_purchases,

                    ROUND(COALESCE(
                        CAST(COUNT(*) FILTER (WHERE ee.action = 'buy' AND ee.discount_pct > 0) AS DOUBLE) /
                        NULLIF(COUNT(*) FILTER (WHERE ee.action = 'buy'), 0),
                        0.0
                    ), 4) AS pct_purchases_with_discount,

                    ROUND(COALESCE(MAX(ee.norm_price) - MIN(ee.norm_price), 0.0), 2) AS price_range_browsed,

                    -- CATEGORY PREFERENCES
                    MODE(ee.category_l1) AS top_category_l1,
                    COUNT(DISTINCT ee.category_l1) AS category_diversity,

                    -- TEMPORAL ENGAGEMENT DISTRIBUTIONS
                    ROUND(COALESCE(
                        CAST(COUNT(*) FILTER (WHERE EXTRACT(DOW FROM ee.timestamp) IN (0, 6)) AS FLOAT) /
                        NULLIF(COUNT(*), 0),
                        0.0
                    ), 4) AS weekend_event_share,

                    ROUND(COALESCE(
                        CAST(COUNT(*) FILTER (WHERE EXTRACT(HOUR FROM ee.timestamp) >= 18) AS FLOAT) /
                        NULLIF(COUNT(*), 0),
                        0.0
                    ), 4) AS evening_event_share

                FROM eligible_events ee
                JOIN user_app_totals uat ON ee.user_id = uat.user_id
                JOIN entropy_calc ec ON ee.user_id = ec.user_id
                LEFT JOIN session_stats ss ON ee.user_id = ss.user_id
                LEFT JOIN session_gap_stats sgs ON ee.user_id = sgs.user_id
                LEFT JOIN cross_app_counts cac ON ee.user_id = cac.user_id
                LEFT JOIN informed_stats ist ON ee.user_id = ist.user_id
                GROUP BY
                    ee.user_id,
                    uat.num_apps_used,
                    ec.app_entropy,
                    ec.dominant_app_share,
                    cac.cross_app_comparison_count,
                    cac.avg_abs_price_gap_pct,
                    ist.informed_purchase_count,
                    ist.chose_cheaper_app_rate,
                    ss.avg_session_duration_minutes,
                    sgs.avg_time_between_sessions_hours
            )
            SELECT * FROM user_features ORDER BY user_id
        """
        result = con.execute(query, params) if params is not None else con.execute(query)
        arrow_table = result.to_arrow_table()
        df = pl.from_arrow(arrow_table)
        return df
    finally:
        if should_close:
            con.close()


def save_behavioral_features(df: pl.DataFrame, output_path: Optional[Union[str, Path]] = None) -> Path:
    """Saves behavioral features to Parquet."""
    out_file = Path(output_path) if output_path else DEFAULT_BEHAVIORAL_PATH
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_file)
    return out_file


def print_comparison_shoppers(df: pl.DataFrame, top_n: int = 5) -> None:
    """
    Ranks and prints the top N users with highest app entropy (comparison shoppers)
    along with their cross-app metrics.
    """
    top_shoppers = df.sort("app_entropy", descending=True).head(top_n)

    print("\n--- Top Comparison-Shopping Users (Highest App Entropy) ---")
    print(f"{'User ID':<10} | {'App Entropy':<12} | {'Dominant App %':<16} | {'Cross-App Comparisons':<22} | {'Num Apps':<8}")
    print("-" * 75)
    for row in top_shoppers.to_dicts():
        print(
            f"{row['user_id']:<10} | "
            f"{row['app_entropy']:<12.4f} | "
            f"{row['dominant_app_share'] * 100:<15.1f}% | "
            f"{row['cross_app_comparison_count']:<22} | "
            f"{row['num_apps_used']:<8}"
        )


def run_behavioral_pipeline(
    reference_date: Optional[str] = None,
    output_path: Optional[str] = None,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Executes the behavioral feature engineering pipeline."""
    db_mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
    con = db_mgr.get_connection()

    try:
        if reference_date is None:
            max_ts = con.execute("SELECT MAX(timestamp) FROM fact_events").fetchone()[0]
            if max_ts is None:
                raise ValueError("No events found in fact_events to determine reference date.")
            reference_date = str(max_ts)

        print(f"Computing behavioral features relative to reference_date: {reference_date}")
        df = compute_behavioral_features(reference_date=reference_date, con=con)
        out_path = save_behavioral_features(df, output_path=output_path)
        print(f"Saved {df.height} user records ({df.width} columns) to {out_path}")

        print_comparison_shoppers(df, top_n=5)

        return {
            "num_users": df.height,
            "num_features": df.width,
            "parquet_path": str(out_path),
            "reference_date": reference_date,
        }
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute cross-app behavioral and entropy features.")
    parser.add_argument("--ref-date", type=str, default=None, help="Reference date (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path")
    args = parser.parse_args()

    run_behavioral_pipeline(reference_date=args.ref_date, output_path=args.output)

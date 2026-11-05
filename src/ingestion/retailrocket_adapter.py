"""
Adapter mapping the RetailRocket clickstream onto this project's star schema.

RetailRocket is a real e-commerce clickstream: 2.76 million view, add-to-cart, and
transaction events from 1.4 million visitors over 4.5 months of 2015. Its event grammar
maps one-to-one onto `fact_events`, which makes it the closest public substitute for the
multi-app telemetry this pipeline was designed around.

Three structural gaps are worth stating up front, because they bound what can be
validated rather than being defects in either the data or the pipeline:

1. **Single platform.** Every event comes from one retailer, so there is no cross-app
   dimension. App entropy, dominant app share, cross-app comparison counts, and the
   `will_switch_app_next_7d` target are all undefined. The switching model cannot be
   evaluated on this data at all.
2. **No product text.** Item properties are hashed, so there are no titles or brands.
   The Week 1 extraction chain cannot be exercised here; that is benchmarked separately
   against real Amazon metadata in `real_data_benchmark.py`.
3. **Obfuscated prices.** Property 790 carries a numeric price, prefixed with `n`, but
   scaled by an undisclosed constant. Relative magnitudes are meaningful and so monetary
   features and CLV work, but absolute currency amounts are not interpretable.

The warehouse is written to a separate database file so the synthetic warehouse and all
committed model artifacts remain untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union

# Ensure consumer-telemetry root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import polars as pl

from src.config import DATA_DIR
from src.features.database import DatabaseManager

EXTERNAL_DIR = DATA_DIR / "external"
RETAILROCKET_DIR = EXTERNAL_DIR / "retailrocket"
EVENTS_PATH = RETAILROCKET_DIR / "events.csv"
ITEM_PROPERTIES_PATH = RETAILROCKET_DIR / "item_properties_part1.csv"
RETAILROCKET_DB_PATH = EXTERNAL_DIR / "retailrocket.duckdb"

# RetailRocket's event vocabulary mapped onto this project's action vocabulary.
ACTION_MAP = {"view": "view", "addtocart": "cart", "transaction": "buy"}

# The property id carrying price. Numeric property values are stored as a string with an
# "n" prefix and three decimal places, for example "n15360.000".
PRICE_PROPERTY = "790"
CATEGORY_PROPERTY = "categoryid"

# A single synthetic app label. Naming it explicitly keeps the single-platform
# limitation visible in the warehouse rather than hiding it behind a plausible value.
SINGLE_APP_NAME = "retailrocket_web"


def load_events(path: Optional[Union[str, Path]] = None) -> pl.DataFrame:
    """
    Loads the raw clickstream and maps it onto the project's event vocabulary.

    Timestamps arrive as epoch milliseconds and are converted to datetimes.
    """
    source = Path(path) if path else EVENTS_PATH
    if not source.exists():
        raise FileNotFoundError(f"RetailRocket events not found at {source}")

    df = pl.read_csv(source)
    return (
        df.with_columns(
            [
                (pl.col("visitorid").cast(pl.String).str.pad_start(9, "0"))
                .map_elements(lambda v: f"rr_{v}", return_dtype=pl.String)
                .alias("user_id"),
                pl.col("itemid")
                .cast(pl.String)
                .map_elements(lambda v: f"item_{v}", return_dtype=pl.String)
                .alias("canonical_product_id"),
                pl.col("event").replace_strict(ACTION_MAP, default=None).alias("action"),
                (pl.col("timestamp") * 1000).cast(pl.Datetime("us")).alias("event_time"),
            ]
        )
        .filter(pl.col("action").is_not_null())
        .select(["user_id", "canonical_product_id", "action", "event_time"])
    )


def load_item_prices(path: Optional[Union[str, Path]] = None) -> pl.DataFrame:
    """
    Extracts the most recent price per item from the hashed item property table.

    Property values are versioned by timestamp, so the latest observation per item is
    taken. The `n` prefix marks a numeric value and is stripped before casting.
    """
    source = Path(path) if path else ITEM_PROPERTIES_PATH
    if not source.exists():
        raise FileNotFoundError(f"RetailRocket item properties not found at {source}")

    lazy = pl.scan_csv(source)
    return (
        lazy.filter(pl.col("property") == PRICE_PROPERTY)
        .with_columns(
            [
                pl.col("value").str.extract(r"n([0-9.]+)", 1).cast(pl.Float64).alias("price"),
                pl.col("itemid")
                .cast(pl.String)
                .map_elements(lambda v: f"item_{v}", return_dtype=pl.String)
                .alias("canonical_product_id"),
            ]
        )
        .filter(pl.col("price").is_not_null() & (pl.col("price") > 0))
        .sort("timestamp")
        .group_by("canonical_product_id")
        .agg(pl.col("price").last().alias("price"))
        .collect()
    )


def load_item_categories(path: Optional[Union[str, Path]] = None) -> pl.DataFrame:
    """Extracts the most recent category id per item, used as a category proxy."""
    source = Path(path) if path else ITEM_PROPERTIES_PATH
    if not source.exists():
        raise FileNotFoundError(f"RetailRocket item properties not found at {source}")

    lazy = pl.scan_csv(source)
    return (
        lazy.filter(pl.col("property") == CATEGORY_PROPERTY)
        .with_columns(
            [
                pl.col("itemid")
                .cast(pl.String)
                .map_elements(lambda v: f"item_{v}", return_dtype=pl.String)
                .alias("canonical_product_id"),
                pl.col("value").cast(pl.String).alias("category_raw"),
            ]
        )
        .sort("timestamp")
        .group_by("canonical_product_id")
        .agg(pl.col("category_raw").last().alias("category_raw"))
        .collect()
        .with_columns(
            ("cat_" + pl.col("category_raw")).alias("category_l1")
        )
        .select(["canonical_product_id", "category_l1"])
    )


def summarize_activity(events: pl.DataFrame) -> pl.DataFrame:
    """Returns the per-user event count distribution, used to choose a density filter."""
    return events.group_by("user_id").agg(pl.len().alias("events"))


def build_warehouse(
    min_events_per_user: int = 20,
    db_path: Optional[Union[str, Path]] = None,
    events_path: Optional[Union[str, Path]] = None,
    properties_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Populates a star schema warehouse from the real clickstream.

    `min_events_per_user` filters to users with enough history for the 7-day and 30-day
    aggregates to mean anything. The raw population averages under two events per
    visitor, which is typical of real web traffic and far sparser than this pipeline's
    feature definitions assume. The chosen threshold and resulting cohort size are
    returned so the filter is always reported alongside any metric computed from it.
    """
    print("Loading real clickstream...")
    events = load_events(events_path)
    total_events = events.height
    total_users = events["user_id"].n_unique()
    print(f"  {total_events:,} events from {total_users:,} visitors")

    activity = summarize_activity(events)
    print(f"  events per visitor: mean {activity['events'].mean():.2f}, "
          f"median {activity['events'].median():.0f}, max {activity['events'].max():,}")

    dense_users = activity.filter(pl.col("events") >= min_events_per_user)["user_id"]
    print(f"  visitors with >= {min_events_per_user} events: {dense_users.len():,} "
          f"({dense_users.len() / total_users:.2%} of all visitors)")

    events = events.filter(pl.col("user_id").is_in(dense_users.implode()))
    print(f"  retained {events.height:,} events "
          f"({events.height / total_events:.1%} of all events)")

    print("Loading item prices from the hashed property table...")
    prices = load_item_prices(properties_path)
    print(f"  prices recovered for {prices.height:,} items")

    print("Loading item categories...")
    categories = load_item_categories(properties_path)
    print(f"  categories recovered for {categories.height:,} items")

    enriched = (
        events.join(prices, on="canonical_product_id", how="left")
        .join(categories, on="canonical_product_id", how="left")
        .with_columns(
            [
                pl.lit(SINGLE_APP_NAME).alias("app_name"),
                pl.lit("INR").alias("currency"),
                pl.lit(None, dtype=pl.String).alias("session_id"),
                pl.lit(None, dtype=pl.String).alias("brand"),
                pl.lit(None, dtype=pl.String).alias("product_title"),
                pl.lit(None, dtype=pl.String).alias("category_l2"),
            ]
        )
        .with_row_index("row_nr")
        .with_columns(
            ("evt_" + pl.col("row_nr").cast(pl.String).str.pad_start(9, "0")).alias("event_id")
        )
        .drop("row_nr")
    )

    price_coverage = enriched["price"].drop_nulls().len() / max(1, enriched.height)
    print(f"  price coverage across retained events: {price_coverage:.1%}")

    target = Path(db_path) if db_path else RETAILROCKET_DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()

    db_mgr = DatabaseManager(db_path=target)
    db_mgr.init_schema()
    con = db_mgr.get_connection()

    try:
        con.register("rr_events", enriched.to_arrow())

        con.execute("DELETE FROM fact_events")
        con.execute("DELETE FROM dim_products")
        con.execute("DELETE FROM dim_users")

        con.execute(
            """
            INSERT INTO fact_events (
                event_id, user_id, timestamp, app_name, canonical_product_id,
                action, price, currency, session_id
            )
            SELECT
                event_id, user_id, CAST(event_time AS TIMESTAMP), app_name,
                canonical_product_id, action,
                CAST(price AS DECIMAL(12,2)), currency, session_id
            FROM rr_events
            """
        )
        con.execute(
            """
            INSERT INTO dim_products (
                canonical_product_id, brand, product_title,
                category_l1, category_l2, avg_price
            )
            SELECT
                canonical_product_id,
                FIRST(brand), FIRST(product_title),
                FIRST(category_l1), FIRST(category_l2),
                CAST(AVG(price) AS DECIMAL(12,2))
            FROM rr_events
            WHERE canonical_product_id IS NOT NULL
            GROUP BY canonical_product_id
            """
        )
        con.execute(
            """
            INSERT INTO dim_users (user_id, first_seen, primary_app, total_events)
            SELECT
                user_id, MIN(CAST(event_time AS TIMESTAMP)),
                MODE(app_name), COUNT(event_id)
            FROM rr_events
            GROUP BY user_id
            """
        )

        counts = {
            "fact_events": con.execute("SELECT COUNT(*) FROM fact_events").fetchone()[0],
            "dim_products": con.execute("SELECT COUNT(*) FROM dim_products").fetchone()[0],
            "dim_users": con.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0],
        }
        bounds = con.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM fact_events"
        ).fetchone()
        action_mix = con.execute(
            "SELECT action, COUNT(*) FROM fact_events GROUP BY action ORDER BY 2 DESC"
        ).fetchall()
    finally:
        con.close()

    print("\nWarehouse built:")
    for table, count in counts.items():
        print(f"  {table}: {count:,}")
    print(f"  time range: {bounds[0]} -> {bounds[1]}")
    print(f"  action mix: {dict(action_mix)}")

    return {
        "db_path": target,
        "counts": counts,
        "time_range": (str(bounds[0]), str(bounds[1])),
        "action_mix": dict(action_mix),
        "raw_events": total_events,
        "raw_users": total_users,
        "min_events_per_user": min_events_per_user,
        "retained_users": int(dense_users.len()),
        "retained_events": int(events.height),
        "price_coverage": price_coverage,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a star schema warehouse from the RetailRocket clickstream."
    )
    parser.add_argument(
        "--min-events", type=int, default=20,
        help="Minimum events per visitor to retain (default: 20)",
    )
    args = parser.parse_args()

    build_warehouse(min_events_per_user=args.min_events)

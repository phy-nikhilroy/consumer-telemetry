"""
Simulates 30 days of e-commerce telemetry for 50 users across 5 behavioral archetypes.
Routes simulated events through Week 1 extraction pipeline and populates DuckDB warehouse.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

# Ensure consumer-telemetry root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import polars as pl

from src.features.database import DatabaseManager, DEFAULT_DB_PATH
from src.ingestion.pipeline import process
from src.ingestion.synthetic_data import CATALOG, _format_inr

# 5 User Behavioral Archetypes
ARCHETYPES = [
    "loyalist",
    "comparison_shopper",
    "bargain_hunter",
    "impulse_buyer",
    "window_shopper",
]

APPS = ["amazon_in", "flipkart", "walmart", "amazon_us"]


class UserProfile:
    """Represents a simulated shopper profile with behavioral constraints."""

    def __init__(self, user_id: str, archetype: str, primary_app: str):
        self.user_id = user_id
        self.archetype = archetype
        self.primary_app = primary_app


def create_user_profiles(num_users: int = 50, seed: int = 42) -> List[UserProfile]:
    """Generates balanced user profiles assigned to archetypes."""
    random.seed(seed)
    profiles = []
    archetypes_cycle = ARCHETYPES * (num_users // len(ARCHETYPES) + 1)

    for i in range(1, num_users + 1):
        uid = f"user_{i:03d}"
        archetype = archetypes_cycle[i - 1]
        primary_app = random.choice(APPS)
        profiles.append(UserProfile(user_id=uid, archetype=archetype, primary_app=primary_app))

    return profiles


def _generate_noisy_text(catalog_item: dict, app: str, action: str, discount_pct: int, rating: float) -> str:
    """Creates realistic noisy raw OCR text resembling app dumps."""
    brand = catalog_item["brand"]
    title = catalog_item["title"]
    specs = " ".join(catalog_item.get("specs", []))
    currency = "INR" if app in ["amazon_in", "flipkart"] else "USD"

    base_price = catalog_item["base_inr"] if currency == "INR" else catalog_item["base_usd"]
    final_price = base_price * (1.0 - discount_pct / 100.0)

    if currency == "INR":
        price_str = f"₹{_format_inr(final_price)}"
        mrp_str = f"M.R.P: ₹{_format_inr(base_price)}" if discount_pct > 0 else ""
    else:
        price_str = f"${final_price:.2f}"
        mrp_str = f"List: ${base_price:.2f}" if discount_pct > 0 else ""

    action_text = {
        "view": "In stock",
        "cart": "Add to Cart",
        "buy": "Buy Now"
    }.get(action, "In stock")

    components = [
        "Sponsored" if random.random() < 0.25 else "",
        f"{rating:.1f}★",
        f"({random.randint(100, 50000)})",
        brand,
        title,
        specs,
        price_str,
        mrp_str,
        f"({discount_pct}% off)" if discount_pct > 0 else "",
        action_text,
    ]
    return " ".join(c for c in components if c).strip()


_catalog_extraction_cache: Dict[str, dict] = {}


def _get_catalog_extraction(item: dict) -> dict:
    """Extracts canonical product attributes via Week 1 pipeline once per unique catalog item."""
    key = f"{item['brand']} {item['title']} {' '.join(item.get('specs', []))}".strip()
    if key not in _catalog_extraction_cache:
        sample_text = f"{key} $99.99 4.5★ Add to Cart"
        prod = process(sample_text)
        _catalog_extraction_cache[key] = {
            "canonical_name": prod.canonical_name,
            "canonical_product_id": prod.canonical_product_id,
            "brand": prod.brand,
            "product_title": prod.product_title,
            "category_l1": prod.category_l1,
            "category_l2": prod.category_l2,
        }
    return _catalog_extraction_cache[key]


def simulate_events_dataset(
    num_users: int = 50,
    days: int = 30,
    seed: int = 42,
    start_date: Optional[datetime] = None
) -> Tuple[List[Dict[str, Any]], List[UserProfile]]:
    """
    Generates realistic 30-day simulated telemetry dataset for 50 users.
    Yields 10,000+ events modeling 5 behavioral archetypes.
    """
    random.seed(seed)
    profiles = create_user_profiles(num_users=num_users, seed=seed)
    if start_date is None:
        start_date = datetime(2026, 9, 28, 8, 0, 0, tzinfo=timezone.utc)

    all_events: List[Dict[str, Any]] = []
    event_counter = 1

    for profile in profiles:
        for day_offset in range(days):
            current_day = start_date + timedelta(days=day_offset)

            # Determine number of events per day based on archetype
            if profile.archetype == "window_shopper":
                events_today = random.randint(12, 22)
            elif profile.archetype == "loyalist":
                events_today = random.randint(6, 14)
            elif profile.archetype == "comparison_shopper":
                events_today = random.randint(8, 16)
            elif profile.archetype == "bargain_hunter":
                events_today = random.randint(6, 15)
            else:  # impulse_buyer
                events_today = random.randint(5, 12)

            # Generate daytime timestamps
            hour_offsets = sorted([random.uniform(0, 14) for _ in range(events_today)])

            i = 0
            while i < events_today:
                ts = current_day + timedelta(hours=hour_offsets[i])

                # App selection by archetype
                if profile.archetype == "loyalist":
                    app = profile.primary_app if random.random() < 0.80 else random.choice(APPS)
                elif profile.archetype == "comparison_shopper":
                    # Random app selection across 3+ apps
                    app = random.choice(APPS)
                else:
                    app = random.choice(APPS)

                # Product selection & discounts
                item = random.choice(CATALOG)
                discount_pct = random.choice([0, 10, 15, 20, 35, 45, 50])
                rating = round(random.uniform(3.5, 4.9), 1)

                # Archetype-specific action probability
                if profile.archetype == "loyalist":
                    action = random.choices(["view", "cart", "buy"], weights=[0.50, 0.25, 0.25])[0]
                elif profile.archetype == "comparison_shopper":
                    action = random.choices(["view", "cart", "buy"], weights=[0.85, 0.12, 0.03])[0]
                elif profile.archetype == "bargain_hunter":
                    if discount_pct >= 30:
                        action = random.choices(["view", "cart", "buy"], weights=[0.30, 0.35, 0.35])[0]
                    else:
                        action = random.choices(["view", "cart", "buy"], weights=[0.95, 0.04, 0.01])[0]
                elif profile.archetype == "impulse_buyer":
                    action = random.choices(["view", "cart", "buy"], weights=[0.40, 0.25, 0.35])[0]
                else:  # window_shopper
                    action = random.choices(["view", "cart", "buy"], weights=[0.96, 0.03, 0.01])[0]

                currency = "INR" if app in ["amazon_in", "flipkart"] else "USD"
                base_price = item["base_inr"] if currency == "INR" else item["base_usd"]
                final_price = round(base_price * (1.0 - discount_pct / 100.0), 2)

                raw_text = _generate_noisy_text(item, app, action, discount_pct, rating)
                extracted = _get_catalog_extraction(item)

                event_record = {
                    "event_id": f"evt_{event_counter:07d}",
                    "user_id": profile.user_id,
                    "archetype": profile.archetype,
                    "timestamp": ts.isoformat(),
                    "app_name": app,
                    "canonical_product_id": extracted["canonical_product_id"],
                    "canonical_name": extracted["canonical_name"],
                    "brand": extracted["brand"],
                    "product_title": extracted["product_title"],
                    "category_l1": extracted["category_l1"],
                    "category_l2": extracted["category_l2"],
                    "action": action,
                    "price": final_price,
                    "currency": currency,
                    "session_id": None,
                }
                all_events.append(event_record)
                event_counter += 1

                # If comparison shopper, simulate multi-app comparison within 1 hour
                if profile.archetype == "comparison_shopper" and random.random() < 0.35 and (i + 1) < events_today:
                    i += 1
                    other_apps = [a for a in APPS if a != app]
                    comp_app = random.choice(other_apps)
                    comp_ts = ts + timedelta(minutes=random.uniform(5, 30))
                    comp_curr = "INR" if comp_app in ["amazon_in", "flipkart"] else "USD"
                    comp_base = item["base_inr"] if comp_curr == "INR" else item["base_usd"]
                    comp_price = round(comp_base * (1.0 - discount_pct / 100.0), 2)
                    comp_text = _generate_noisy_text(item, comp_app, "view", discount_pct, rating)

                    comp_record = {
                        "event_id": f"evt_{event_counter:07d}",
                        "user_id": profile.user_id,
                        "archetype": profile.archetype,
                        "timestamp": comp_ts.isoformat(),
                        "app_name": comp_app,
                        "canonical_product_id": extracted["canonical_product_id"],
                        "canonical_name": extracted["canonical_name"],
                        "brand": extracted["brand"],
                        "product_title": extracted["product_title"],
                        "category_l1": extracted["category_l1"],
                        "category_l2": extracted["category_l2"],
                        "action": "view",
                        "price": comp_price,
                        "currency": comp_curr,
                        "session_id": None,
                    }
                    all_events.append(comp_record)
                    event_counter += 1

                i += 1

    return all_events, profiles


def reload_duckdb_warehouse(
    events: List[Dict[str, Any]],
    profiles: List[UserProfile],
    db_path: Optional[str] = None
) -> Dict[str, int]:
    """
    Populates DuckDB warehouse tables (fact_events, dim_products, dim_users)
    with the expanded simulated dataset.
    """
    db_mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
    db_mgr.init_schema()

    df = pl.DataFrame(events)

    con = db_mgr.get_connection()
    try:
        con.register("expanded_events_view", df.to_arrow())

        # Clean existing tables and reload
        con.execute("DELETE FROM fact_events")
        con.execute("DELETE FROM dim_products")
        con.execute("DELETE FROM dim_users")

        # 1. Insert fact_events
        con.execute("""
            INSERT INTO fact_events (
                event_id,
                user_id,
                timestamp,
                app_name,
                canonical_product_id,
                action,
                price,
                currency,
                session_id
            )
            SELECT
                event_id,
                user_id,
                CAST(timestamp AS TIMESTAMP) AS timestamp,
                app_name,
                canonical_product_id,
                action,
                CAST(price AS DECIMAL(10,2)) AS price,
                currency,
                session_id
            FROM expanded_events_view
        """)

        # 2. Insert dim_products
        con.execute("""
            INSERT INTO dim_products (
                canonical_product_id,
                brand,
                product_title,
                category_l1,
                category_l2,
                avg_price
            )
            SELECT
                canonical_product_id,
                FIRST(brand) FILTER (WHERE brand IS NOT NULL) AS brand,
                FIRST(canonical_name) FILTER (WHERE canonical_name IS NOT NULL) AS product_title,
                FIRST(category_l1) FILTER (WHERE category_l1 IS NOT NULL) AS category_l1,
                FIRST(category_l2) FILTER (WHERE category_l2 IS NOT NULL) AS category_l2,
                CAST(AVG(price) AS DECIMAL(10,2)) AS avg_price
            FROM expanded_events_view
            WHERE canonical_product_id IS NOT NULL
            GROUP BY canonical_product_id
        """)

        # 3. Insert dim_users
        con.execute("""
            INSERT INTO dim_users (
                user_id,
                first_seen,
                primary_app,
                total_events
            )
            SELECT
                user_id,
                MIN(CAST(timestamp AS TIMESTAMP)) AS first_seen,
                MODE(app_name) AS primary_app,
                COUNT(event_id) AS total_events
            FROM expanded_events_view
            GROUP BY user_id
        """)

        fact_count = con.execute("SELECT COUNT(*) FROM fact_events").fetchone()[0]
        prod_count = con.execute("SELECT COUNT(*) FROM dim_products").fetchone()[0]
        user_count = con.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0]

        return {
            "fact_events": fact_count,
            "dim_products": prod_count,
            "dim_users": user_count,
        }
    finally:
        con.close()


def run_simulation(
    num_users: int = 50,
    days: int = 30,
    db_path: Optional[str] = None,
    output_parquet: str = "data/processed/simulated_events.parquet"
) -> Dict[str, Any]:
    """Generates dataset and reloads DuckDB."""
    print(f"Simulating {num_users} users over {days} days...")
    events, profiles = simulate_events_dataset(num_users=num_users, days=days)
    print(f"Generated {len(events)} events across 5 archetypes.")

    # Save to parquet
    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(events)
    df.write_parquet(out_path)
    print(f"Saved simulated events to {out_path} ({df.shape[0]} rows).")

    # Reload DuckDB
    counts = reload_duckdb_warehouse(events, profiles, db_path=db_path)
    print("Warehouse reloaded:")
    for table, count in counts.items():
        print(f"  {table}: {count} records")

    return {
        "total_events": len(events),
        "parquet_path": str(out_path),
        "warehouse_counts": counts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate user shopping telemetry across archetypes.")
    parser.add_argument("--users", type=int, default=50, help="Number of users to simulate (default: 50)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to simulate (default: 30)")
    args = parser.parse_args()

    run_simulation(num_users=args.users, days=args.days)

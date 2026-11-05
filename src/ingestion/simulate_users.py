"""
Simulates 30 days of e-commerce telemetry across 5 behavioral archetypes.

Each simulated user carries latent traits (activity level, purchase propensity, price
sensitivity, app loyalty, and an optional dormancy onset) drawn once at profile creation.
Those traits drive both the historical events that features are computed from and the
forward events that targets are computed from, so the engineered feature store carries
genuine predictive signal rather than per-archetype noise.

Routes simulated events through the Week 1 extraction pipeline and populates the DuckDB
warehouse.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure consumer-telemetry root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import polars as pl

from src.features.database import DatabaseManager, DEFAULT_DB_PATH
from src.ingestion.pipeline import process
from src.config import USD_TO_INR_RATE
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

# Per-archetype generative parameters.
#
# buy_rate is the baseline per-event probability of a "buy" action. The values are
# deliberately low: at roughly 10 events per day over 30 days, a buy_rate of 0.027
# yields about 8 purchases per user per month, which keeps the forward 7-day purchase
# target in a learnable range instead of saturating at nearly 100 percent positive.
#
# cart_rate is the baseline per-event probability of a "cart" action.
# events_per_day bounds the daily event volume before the per-user activity multiplier.
# app_loyalty is the probability of staying on the user's primary app.
# discount_sensitivity scales how strongly a deep discount lifts conversion.
ARCHETYPE_PARAMS: Dict[str, Dict[str, Any]] = {
    "loyalist": {
        "events_per_day": (6, 14),
        "buy_rate": 0.027,
        "cart_rate": 0.110,
        "app_loyalty": 0.80,
        "discount_sensitivity": 0.8,
    },
    "impulse_buyer": {
        "events_per_day": (5, 12),
        "buy_rate": 0.018,
        "cart_rate": 0.085,
        "app_loyalty": 0.45,
        "discount_sensitivity": 1.0,
    },
    "bargain_hunter": {
        "events_per_day": (6, 15),
        "buy_rate": 0.010,
        "cart_rate": 0.060,
        "app_loyalty": 0.35,
        "discount_sensitivity": 3.0,
    },
    "comparison_shopper": {
        "events_per_day": (8, 16),
        "buy_rate": 0.0033,
        "cart_rate": 0.040,
        "app_loyalty": 0.28,
        "discount_sensitivity": 1.6,
    },
    "window_shopper": {
        "events_per_day": (12, 22),
        "buy_rate": 0.0005,
        "cart_rate": 0.022,
        "app_loyalty": 0.30,
        "discount_sensitivity": 0.5,
    },
}

# Fraction of the population that goes dormant partway through the observation window.
#
# This is the only source of churn signal in the dataset. BG/NBD infers a customer's
# dropout probability from how long before the observation end their last purchase sits,
# so a population where almost everyone stays active to the final day makes the dropout
# parameter unidentifiable and collapses P(alive) to 1.0 for every user. A quarter of the
# population going dormant, with onsets spread from the second week onward, gives the
# model a real signal to fit.
DORMANCY_SHARE = 0.25
DORMANCY_ONSET_RANGE = (8, 24)

# Price tiers in INR, used to give each shopper a consistent spending level.
#
# The catalog spans roughly 1,100 to 130,000 INR. Drawing from it uniformly gives every
# user the same wildly dispersed order value distribution, so within-customer variance
# swamps between-customer variance. Gamma-Gamma models each customer as having a stable
# mean order value, and cannot be identified without one. Assigning each user a tier
# affinity makes spend a persistent customer trait, as it is in reality: a budget
# accessories buyer and a flagship phone buyer are different customers, not the same
# customer on different days.
PRICE_TIERS: Dict[str, Tuple[float, float]] = {
    "budget": (0.0, 10_000.0),
    "mid": (10_000.0, 50_000.0),
    "premium": (50_000.0, float("inf")),
}

# Probability that a user picks a product from their own price tier.
TIER_AFFINITY = 0.78


def _catalog_by_tier() -> Dict[str, List[dict]]:
    """Buckets the product catalog into price tiers by INR list price."""
    buckets: Dict[str, List[dict]] = {name: [] for name in PRICE_TIERS}
    for item in CATALOG:
        price = float(item["base_inr"])
        for name, (low, high) in PRICE_TIERS.items():
            if low <= price < high:
                buckets[name].append(item)
                break
    # Never leave a tier empty, or users assigned to it would have no products.
    return {name: items or list(CATALOG) for name, items in buckets.items()}


CATALOG_BY_TIER: Dict[str, List[dict]] = _catalog_by_tier()


class UserProfile:
    """
    Represents a simulated shopper profile with behavioral constraints and latent traits.

    The latent traits are the mechanism that makes downstream supervised learning
    meaningful: they persist for the whole observation window, so a user's historical
    behavior (which features summarize) and their future behavior (which targets
    measure) are driven by the same underlying parameters.
    """

    def __init__(
        self,
        user_id: str,
        archetype: str,
        primary_app: str,
        activity_level: float = 1.0,
        buy_propensity: float = 1.0,
        price_sensitivity: float = 1.0,
        app_loyalty: float = 0.5,
        dormant_from_day: Optional[int] = None,
        price_tier: str = "mid",
    ) -> None:
        self.user_id = user_id
        self.archetype = archetype
        self.primary_app = primary_app
        # Multiplier on daily event volume.
        self.activity_level = activity_level
        # Multiplier on the archetype baseline buy and cart rates.
        self.buy_propensity = buy_propensity
        # Multiplier on the archetype discount sensitivity.
        self.price_sensitivity = price_sensitivity
        # Probability of choosing the primary app for any given event.
        self.app_loyalty = app_loyalty
        # Day offset after which the user stops generating events, or None if always active.
        self.dormant_from_day = dormant_from_day
        # Preferred catalog price band, which keeps order values stable per customer.
        self.price_tier = price_tier

    def choose_product(self) -> dict:
        """
        Selects a catalog item, favouring the user's own price tier.

        The occasional out-of-tier pick keeps some within-customer spend variance, which
        the Gamma-Gamma model needs in order to have anything to estimate.
        """
        if random.random() < TIER_AFFINITY:
            return random.choice(CATALOG_BY_TIER[self.price_tier])
        return random.choice(CATALOG)

    @property
    def params(self) -> Dict[str, Any]:
        """Returns the generative parameter block for this profile's archetype."""
        return ARCHETYPE_PARAMS[self.archetype]

    def is_active_on_day(self, day_offset: int) -> bool:
        """Returns False once the user has gone dormant."""
        return self.dormant_from_day is None or day_offset < self.dormant_from_day

    def effective_buy_rate(self, discount_pct: int) -> float:
        """
        Returns this user's per-event purchase probability for an item at a given discount.

        Combines the archetype baseline, the user's latent propensity multiplier, and a
        discount uplift scaled by the user's price sensitivity.
        """
        params = self.params
        base = params["buy_rate"] * self.buy_propensity
        sensitivity = params["discount_sensitivity"] * self.price_sensitivity
        uplift = 1.0 + sensitivity * (discount_pct / 100.0)
        return min(0.60, base * uplift)

    def effective_cart_rate(self, discount_pct: int) -> float:
        """Returns this user's per-event cart-add probability at a given discount."""
        params = self.params
        base = params["cart_rate"] * (0.5 + 0.5 * self.buy_propensity)
        sensitivity = params["discount_sensitivity"] * self.price_sensitivity
        uplift = 1.0 + 0.5 * sensitivity * (discount_pct / 100.0)
        return min(0.75, base * uplift)


def create_user_profiles(num_users: int = 400, seed: int = 42) -> List[UserProfile]:
    """
    Generates balanced user profiles assigned to archetypes, each with latent traits.

    Traits are drawn from log-normal distributions so the population contains a realistic
    long tail of highly active and highly convertive users rather than a single
    homogeneous block per archetype.
    """
    random.seed(seed)
    profiles: List[UserProfile] = []
    archetypes_cycle = ARCHETYPES * (num_users // len(ARCHETYPES) + 1)

    dormant_count = int(num_users * DORMANCY_SHARE)
    dormant_ids = set(random.sample(range(1, num_users + 1), dormant_count)) if dormant_count else set()

    for i in range(1, num_users + 1):
        uid = f"user_{i:03d}"
        archetype = archetypes_cycle[i - 1]
        params = ARCHETYPE_PARAMS[archetype]

        primary_app = random.choice(APPS)

        # Latent traits. Log-normal keeps every multiplier strictly positive and
        # right-skewed, clamped to avoid degenerate extremes.
        activity_level = min(2.2, max(0.45, random.lognormvariate(0.0, 0.32)))
        buy_propensity = min(2.6, max(0.25, random.lognormvariate(0.0, 0.52)))
        price_sensitivity = min(2.5, max(0.40, random.lognormvariate(0.0, 0.38)))

        # Loyalty jitters around the archetype baseline.
        app_loyalty = min(0.95, max(0.25, params["app_loyalty"] + random.uniform(-0.10, 0.10)))

        dormant_from_day = random.randint(*DORMANCY_ONSET_RANGE) if i in dormant_ids else None

        # Spend tier is a persistent customer trait. Bargain hunters skew budget and
        # loyalists skew premium, but every archetype spans all three tiers.
        if archetype == "bargain_hunter":
            tier_weights = [0.60, 0.30, 0.10]
        elif archetype == "loyalist":
            tier_weights = [0.20, 0.40, 0.40]
        elif archetype == "window_shopper":
            tier_weights = [0.25, 0.35, 0.40]
        else:
            tier_weights = [0.34, 0.40, 0.26]
        price_tier = random.choices(list(PRICE_TIERS.keys()), weights=tier_weights)[0]

        profiles.append(
            UserProfile(
                user_id=uid,
                archetype=archetype,
                primary_app=primary_app,
                activity_level=activity_level,
                buy_propensity=buy_propensity,
                price_sensitivity=price_sensitivity,
                app_loyalty=app_loyalty,
                dormant_from_day=dormant_from_day,
                price_tier=price_tier,
            )
        )

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
    num_users: int = 400,
    days: int = 30,
    seed: int = 42,
    start_date: Optional[datetime] = None
) -> Tuple[List[Dict[str, Any]], List[UserProfile]]:
    """
    Generates a realistic multi-day simulated telemetry dataset.

    Daily event volume, app choice, and conversion are all driven by the latent traits
    held on each UserProfile, with a weekend activity uplift and an optional dormancy
    onset per user.
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

            # Dormant users stop generating events entirely from their onset day.
            if not profile.is_active_on_day(day_offset):
                continue

            # Daily event volume: archetype band, scaled by the user's latent activity
            # level, with a weekend uplift so timing features carry real variation.
            low, high = profile.params["events_per_day"]
            weekend_multiplier = 1.25 if current_day.weekday() >= 5 else 1.0
            events_today = int(round(random.randint(low, high) * profile.activity_level * weekend_multiplier))
            events_today = max(1, events_today)

            # Generate daytime timestamps
            hour_offsets = sorted([random.uniform(0, 14) for _ in range(events_today)])

            i = 0
            while i < events_today:
                ts = current_day + timedelta(hours=hour_offsets[i])

                # App selection driven by the user's latent loyalty.
                if random.random() < profile.app_loyalty:
                    app = profile.primary_app
                else:
                    app = random.choice([a for a in APPS if a != profile.primary_app])

                # Product selection & discounts
                item = profile.choose_product()
                discount_pct = random.choice([0, 10, 15, 20, 35, 45, 50])
                rating = round(random.uniform(3.5, 4.9), 1)

                # Action probability from the user's own effective conversion rates,
                # which fold in archetype baseline, latent propensity, and discount depth.
                buy_p = profile.effective_buy_rate(discount_pct)
                cart_p = profile.effective_cart_rate(discount_pct)
                view_p = max(0.0, 1.0 - buy_p - cart_p)
                action = random.choices(["view", "cart", "buy"], weights=[view_p, cart_p, buy_p])[0]

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
                comparison_chance = 0.35 if profile.archetype == "comparison_shopper" else 0.04
                if random.random() < comparison_chance and (i + 1) < events_today:
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
        con.execute(f"""
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
                -- Normalize currency before averaging. Averaging raw prices across a
                -- mix of INR and USD listings understates the true average by a factor
                -- of roughly two, and because the behavioral feature layer compares a
                -- currency-normalized price against this column, an unnormalized value
                -- made almost every purchase look like it was above average price,
                -- silently zeroing out the discount features for 96 percent of users.
                CAST(AVG(
                    CASE WHEN currency = 'USD' THEN price * {USD_TO_INR_RATE} ELSE price END
                ) AS DECIMAL(12,2)) AS avg_price
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
    num_users: int = 400,
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
    parser.add_argument("--users", type=int, default=400, help="Number of users to simulate (default: 400)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to simulate (default: 30)")
    args = parser.parse_args()

    run_simulation(num_users=args.users, days=args.days)

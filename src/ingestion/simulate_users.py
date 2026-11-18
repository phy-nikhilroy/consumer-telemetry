"""
Simulates 30 days of e-commerce telemetry across 5 behavioral archetypes.

Each simulated user carries latent traits (activity level, purchase propensity, price
sensitivity, app loyalty, and an optional dormancy onset) drawn once at profile creation.
Those traits drive both the historical events that features are computed from and the
forward events that targets are computed from, so the engineered feature store carries
genuine predictive signal rather than per-archetype noise.

Every event is emitted as raw screen text: the focal listing plus 0 to
`SIMULATION_MAX_CONTEXT_LISTINGS` listings of the same category, rendered with
`screen_generator` and OCR noise at `--ocr-noise`. The warehouse is filled only by the raw
event loader (`src/features/event_loader.py`), so the models see real extraction output.
What the simulator intended (archetype, product, action, price, currency, app) is written
to a separate truth file read only by evaluation code, never to the warehouse.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure consumer-telemetry root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.features.database import DatabaseManager, DEFAULT_DB_PATH
from src.config import (
    PRICE_TIERS,
    RANDOM_SEED,
    SIMULATION_LOAD_WORKERS,
    SIMULATION_MAX_CONTEXT_LISTINGS,
    SIMULATION_OCR_NOISE,
    SIMULATION_TRUTH_PATH,
)
from src.ingestion.screen_generator import apply_ocr_noise, render_listing, render_screen
from src.ingestion.synthetic_data import CATALOG, generate_app_screen_text

# Columns of the raw event frame the loader reads, and of the truth frame evaluation reads.
RAW_EVENT_COLUMNS = ("event_id", "user_id", "timestamp", "app_name", "raw_text")
TRUTH_COLUMNS = (
    "event_id", "user_id", "archetype", "true_sku_key", "true_action",
    "true_price", "true_currency", "true_app",
)
# The screen renderer draws from its own generator, seeded from the simulation seed, so
# rendering never shifts the behavioural draws on the module generator.
SCREEN_SEED_OFFSET = 1_000_003

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


def _consume_v1_text_draws(catalog_item: dict, app: str, action: str, discount_pct: int, rating: float) -> None:
    """
    Makes the module-generator draws the v1.0 single-line screen text made.

    The text itself is no longer used. Keeping the draws keeps the behavioural stream (users,
    timestamps, apps, products, actions, prices, and event ids) identical to the v1.2
    simulator for the same seed, so a change in any model number reflects extraction, not a
    different population. The v1.4 behaviour engine replaces this module.
    """
    generate_app_screen_text(catalog_item, app, action, discount_pct, rating)


def sku_key(item: dict) -> str:
    """The ground-truth identity of a catalog item: brand, title, and specs."""
    return " ".join([item["brand"], item["title"], *item.get("specs", [])]).strip()


def _context_items(item: dict, rng: random.Random) -> List[dict]:
    """0 to SIMULATION_MAX_CONTEXT_LISTINGS other catalog items of the focal item's category."""
    pool = [c for c in CATALOG if c is not item and c.get("category") == item.get("category")]
    count = min(len(pool), rng.randint(0, SIMULATION_MAX_CONTEXT_LISTINGS))
    return rng.sample(pool, count)


def render_event_screen(
    item: dict,
    app: str,
    action: str,
    discount_pct: int,
    rating: float,
    price: float,
    rng: random.Random,
    ocr_noise: float = SIMULATION_OCR_NOISE,
) -> str:
    """
    Renders one event's screen: the focal listing first, at the simulated `price`, then
    context listings of the same category with their own prices and a view action, then OCR
    noise at `ocr_noise`. Context listings never carry buy or cart, so the extractor's focal
    rule (first buy or cart listing, otherwise the first listing) picks the focal one.
    """
    listings = [render_listing(item, app, action, discount_pct, rating, rng, price=price, title_variant=False)]
    for other in _context_items(item, rng):
        listings.append(render_listing(
            other, app, "view", rng.choice([0, 10, 20]), round(rng.uniform(3.5, 4.9), 1), rng,
            title_variant=False,
        ))
    return apply_ocr_noise(render_screen(listings, rng), ocr_noise, rng)


def simulate_events_dataset(
    num_users: int = 400,
    days: int = 30,
    seed: int = 42,
    start_date: Optional[datetime] = None,
    ocr_noise: float = SIMULATION_OCR_NOISE,
) -> Tuple[List[Dict[str, Any]], List[UserProfile]]:
    """
    Generates a realistic multi-day simulated telemetry dataset.

    Each event dict carries the raw event (`RAW_EVENT_COLUMNS`) and its ground truth
    (`TRUTH_COLUMNS`). `split_events` separates the two.

    Daily event volume, app choice, and conversion are all driven by the latent traits
    held on each UserProfile, with a weekend activity uplift and an optional dormancy
    onset per user.
    """
    random.seed(seed)
    screen_rng = random.Random(seed + SCREEN_SEED_OFFSET)
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

                if currency == "INR":
                    # INR screens show whole rupees, so the true price is the shown one.
                    final_price = float(round(final_price))
                _consume_v1_text_draws(item, app, action, discount_pct, rating)

                event_record = {
                    "event_id": f"evt_{event_counter:07d}",
                    "user_id": profile.user_id,
                    "timestamp": ts.isoformat(),
                    "app_name": app,
                    "raw_text": render_event_screen(
                        item, app, action, discount_pct, rating, final_price, screen_rng, ocr_noise
                    ),
                    "archetype": profile.archetype,
                    "true_sku_key": sku_key(item),
                    "true_action": action,
                    "true_price": final_price,
                    "true_currency": currency,
                    "true_app": app,
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
                    if comp_curr == "INR":
                        comp_price = float(round(comp_price))
                    _consume_v1_text_draws(item, comp_app, "view", discount_pct, rating)

                    comp_record = {
                        "event_id": f"evt_{event_counter:07d}",
                        "user_id": profile.user_id,
                        "timestamp": comp_ts.isoformat(),
                        "app_name": comp_app,
                        "raw_text": render_event_screen(
                            item, comp_app, "view", discount_pct, rating, comp_price, screen_rng, ocr_noise
                        ),
                        "archetype": profile.archetype,
                        "true_sku_key": sku_key(item),
                        "true_action": "view",
                        "true_price": comp_price,
                        "true_currency": comp_curr,
                        "true_app": comp_app,
                    }
                    all_events.append(comp_record)
                    event_counter += 1

                i += 1

    return all_events, profiles


def split_events(events: List[Dict[str, Any]]) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Splits simulated events into the raw event frame and the ground-truth frame."""
    frame = pl.DataFrame(events) if events else pl.DataFrame(
        schema={c: pl.Utf8 for c in RAW_EVENT_COLUMNS + TRUTH_COLUMNS[2:]}
    )
    raw = frame.select(RAW_EVENT_COLUMNS).with_columns(
        pl.col("timestamp").str.to_datetime(time_zone="UTC").dt.replace_time_zone(None)
        if frame.schema["timestamp"] == pl.Utf8 else pl.col("timestamp")
    )
    return raw, frame.select(TRUTH_COLUMNS)


def reload_duckdb_warehouse(
    events: List[Dict[str, Any]],
    profiles: List[UserProfile],
    db_path: Optional[str] = None,
    workers: int = SIMULATION_LOAD_WORKERS,
) -> Dict[str, int]:
    """
    Rebuilds the warehouse at `db_path` from the simulated raw events.

    Every warehouse table is emptied, then the raw screens go through `load_raw_events`,
    which extracts them and fills `raw_events`, `fact_events`, `fact_price_observations`,
    `dim_products`, and `dim_users`. No ground-truth column reaches the warehouse.
    """
    from src.features.event_loader import load_raw_events

    db_mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
    db_mgr.init_schema()
    raw, _ = split_events(events)

    con = db_mgr.get_connection()
    try:
        for table in ("fact_cross_app_comparisons", "fact_price_observations", "fact_events",
                      "raw_events", "product_aliases", "dim_products", "dim_users"):
            con.execute(f"DELETE FROM {table}")
        load_raw_events(con, raw, workers=workers)
        return {
            table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("raw_events", "fact_events", "fact_price_observations", "dim_products", "dim_users")
        }
    finally:
        con.close()


def default_truth_path(output_parquet: Union[str, Path]) -> Path:
    """
    Where the truth file goes for a given raw event parquet: `SIMULATION_TRUTH_PATH`'s name
    beside the default `simulated_events.parquet`, and `<stem>_truth.parquet` beside any
    other, so an isolated run never overwrites the default truth file.
    """
    out = Path(output_parquet)
    if out.name == "simulated_events.parquet":
        return out.with_name(SIMULATION_TRUTH_PATH.name)
    return out.with_name(f"{out.stem}_truth.parquet")


def run_simulation(
    num_users: int = 400,
    days: int = 30,
    db_path: Optional[str] = None,
    output_parquet: str = "data/processed/simulated_events.parquet",
    seed: int = RANDOM_SEED,
    ocr_noise: float = SIMULATION_OCR_NOISE,
    workers: int = SIMULATION_LOAD_WORKERS,
    truth_parquet: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Simulates `seed`, writes the raw events to `output_parquet` and the ground truth to
    `truth_parquet` (default `default_truth_path(output_parquet)`), and rebuilds the
    warehouse at `db_path` through the raw event loader with `workers` extraction processes.
    """
    print(f"Simulating {num_users} users over {days} days (OCR noise {ocr_noise})...")
    events, profiles = simulate_events_dataset(num_users=num_users, days=days, seed=seed, ocr_noise=ocr_noise)
    print(f"Generated {len(events)} events across 5 archetypes.")

    raw, truth = split_events(events)
    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw.write_parquet(out_path)
    truth_path = Path(truth_parquet) if truth_parquet else default_truth_path(out_path)
    truth.write_parquet(truth_path)
    print(f"Saved raw events to {out_path} and ground truth to {truth_path} ({raw.height} rows).")

    counts = reload_duckdb_warehouse(events, profiles, db_path=db_path, workers=workers)
    print("Warehouse reloaded:")
    for table, count in counts.items():
        print(f"  {table}: {count} records")

    return {
        "total_events": len(events),
        "parquet_path": str(out_path),
        "truth_path": str(truth_path),
        "warehouse_counts": counts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate user shopping telemetry across archetypes.")
    parser.add_argument("--users", type=int, default=400, help="Number of users to simulate (default: 400)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to simulate (default: 30)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Simulation seed (default: RANDOM_SEED)")
    parser.add_argument("--db-path", type=str, default=None, help="Warehouse to rebuild (default: batch warehouse)")
    parser.add_argument("--output-parquet", type=str, default="data/processed/simulated_events.parquet",
                        help="Where to write the simulated raw events")
    parser.add_argument("--truth-parquet", type=str, default=None,
                        help="Where to write the ground truth (default: beside --output-parquet)")
    parser.add_argument("--ocr-noise", type=float, default=SIMULATION_OCR_NOISE,
                        help=f"OCR character noise rate of the rendered screens (default: {SIMULATION_OCR_NOISE})")
    parser.add_argument("--workers", type=int, default=SIMULATION_LOAD_WORKERS,
                        help=f"Extraction processes for the load (default: {SIMULATION_LOAD_WORKERS})")
    args = parser.parse_args()

    run_simulation(num_users=args.users, days=args.days, db_path=args.db_path,
                   output_parquet=args.output_parquet, seed=args.seed, ocr_noise=args.ocr_noise,
                   workers=args.workers, truth_parquet=args.truth_parquet)

"""
Command line front end of the behaviour engine: simulate, write raw and truth files, load.

The shoppers, sessions, funnel, comparisons, app choice, churn, and archetype drift live in
`src/simulation/behavior.py`, and every price comes from `src/simulation/pricebook.py`. This
module builds the catalog (`build_catalog()`), the price book and the engine for one seed, writes
the raw screens to `output_parquet` and the ground truth to a separate truth file, and rebuilds
the warehouse through the raw event loader (`src/features/event_loader.py`), so the warehouse
only ever holds what extraction produced. The truth file is read only by evaluation code.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure consumer-telemetry root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.config import (
    PRICE_BETA,
    RANDOM_SEED,
    SIMULATION_DAYS,
    SIMULATION_LOAD_WORKERS,
    SIMULATION_OCR_NOISE,
    SIMULATION_START,
    SIMULATION_TRUTH_PATH,
    SIMULATION_USERS,
)
from src.features.database import DEFAULT_DB_PATH, DatabaseManager
from src.ingestion.catalog import build_catalog
from src.simulation.behavior import (
    APPS,
    ARCHETYPES,
    RAW_COLUMNS,
    TRUTH_COLUMNS,
    BehaviorEngine,
    UserProfile,
    events_to_frames,
    sku_key,
)
from src.simulation.pricebook import PriceBook

RAW_EVENT_COLUMNS = RAW_COLUMNS
DEFAULT_OUTPUT_PARQUET = "data/processed/simulated_events.parquet"

__all__ = [
    "APPS", "ARCHETYPES", "RAW_EVENT_COLUMNS", "TRUTH_COLUMNS", "build_engine", "default_truth_path",
    "reload_duckdb_warehouse", "run_simulation", "simulate_events_dataset", "sku_key", "split_events",
]


def build_engine(
    days: int = SIMULATION_DAYS,
    seed: int = RANDOM_SEED,
    ocr_noise: float = SIMULATION_OCR_NOISE,
    price_beta: float = PRICE_BETA,
) -> BehaviorEngine:
    """The engine for one seed over `build_catalog()` and a price book covering `days`."""
    catalog = build_catalog()
    return BehaviorEngine(catalog, PriceBook(catalog, APPS, days, seed), seed=seed,
                          price_beta=price_beta, ocr_noise=ocr_noise)


def simulate_events_dataset(
    num_users: int = SIMULATION_USERS,
    days: int = SIMULATION_DAYS,
    seed: int = RANDOM_SEED,
    start_date: Optional[datetime] = None,
    ocr_noise: float = SIMULATION_OCR_NOISE,
    price_beta: float = PRICE_BETA,
) -> Tuple[List[Dict[str, Any]], List[UserProfile]]:
    """
    Simulates `num_users` shoppers for `days` days from `start_date` (default
    `SIMULATION_START`). Returns the event dicts (raw and truth columns, in timestamp order)
    and the profiles. `split_events` separates raw events from truth.
    """
    start = _naive(start_date) if start_date is not None else datetime.fromisoformat(SIMULATION_START)
    engine = build_engine(days=days, seed=seed, ocr_noise=ocr_noise, price_beta=price_beta)
    profiles = engine.create_profiles(num_users)
    return engine.simulate_events(profiles, start, days), profiles


def _naive(value: datetime) -> datetime:
    """Drops a UTC offset; warehouse timestamps are naive UTC."""
    if value.tzinfo is not None:
        from datetime import timezone

        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def split_events(events: Sequence[Dict[str, Any]]) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Splits simulated event dicts into the raw event frame and the ground-truth frame."""
    return events_to_frames(events)


def reload_duckdb_warehouse(
    events: Sequence[Dict[str, Any]],
    profiles: Sequence[UserProfile],
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
    num_users: int = SIMULATION_USERS,
    days: int = SIMULATION_DAYS,
    db_path: Optional[str] = None,
    output_parquet: str = DEFAULT_OUTPUT_PARQUET,
    seed: int = RANDOM_SEED,
    ocr_noise: float = SIMULATION_OCR_NOISE,
    workers: int = SIMULATION_LOAD_WORKERS,
    truth_parquet: Optional[str] = None,
    price_beta: float = PRICE_BETA,
) -> Dict[str, Any]:
    """
    Simulates `seed`, writes the raw events to `output_parquet` and the ground truth to
    `truth_parquet` (default `default_truth_path(output_parquet)`), and rebuilds the
    warehouse at `db_path` through the raw event loader with `workers` extraction processes.
    """
    print(f"Simulating {num_users} users over {days} days (seed {seed}, OCR noise {ocr_noise}, "
          f"price beta {price_beta})...")
    events, profiles = simulate_events_dataset(num_users=num_users, days=days, seed=seed,
                                               ocr_noise=ocr_noise, price_beta=price_beta)
    print(f"Generated {len(events)} events.")

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
    parser = argparse.ArgumentParser(description="Simulate shoppers with the behaviour engine and load the warehouse.")
    parser.add_argument("--users", type=int, default=SIMULATION_USERS,
                        help=f"Number of users to simulate (default: {SIMULATION_USERS})")
    parser.add_argument("--days", type=int, default=SIMULATION_DAYS,
                        help=f"Number of days to simulate (default: {SIMULATION_DAYS})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Simulation seed (default: RANDOM_SEED)")
    parser.add_argument("--price-beta", type=float, default=PRICE_BETA,
                        help=f"Price coefficient of the purchase-app logit (default: {PRICE_BETA})")
    parser.add_argument("--db-path", type=str, default=None, help="Warehouse to rebuild (default: batch warehouse)")
    parser.add_argument("--output-parquet", type=str, default=DEFAULT_OUTPUT_PARQUET,
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
                   workers=args.workers, truth_parquet=args.truth_parquet, price_beta=args.price_beta)

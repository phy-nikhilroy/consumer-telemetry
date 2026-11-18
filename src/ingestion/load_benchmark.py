"""
Times loading the default simulated population into a fresh warehouse.

The default simulation (400 users, 30 days, seed RANDOM_SEED, SIMULATION_OCR_NOISE) is
generated, then its raw screens are loaded with `load_raw_events` into a new warehouse
under `data/experiments/v1_3_load/`, never the batch warehouse. The extraction wait,
product resolution, and warehouse writes are timed separately, and the result is written
to `reports/v1_3_load.json` for gate G1.3-11.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from src.config import DEFAULT_DB_PATH, RANDOM_SEED, SIMULATION_LOAD_WORKERS, SIMULATION_OCR_NOISE

DEFAULT_OUT = PROJECT_ROOT / "data" / "experiments" / "v1_3_load"


def time_load(
    out_dir: Path = DEFAULT_OUT,
    users: int = 400,
    days: int = 30,
    seed: int = RANDOM_SEED,
    ocr_noise: float = SIMULATION_OCR_NOISE,
    workers: int = SIMULATION_LOAD_WORKERS,
) -> Dict[str, Any]:
    """Simulates, loads into `out_dir/warehouse.duckdb`, and returns counts and timings."""
    from src.features.event_loader import load_raw_events
    from src.features.migrations import migrate
    from src.ingestion.simulate_users import simulate_events_dataset, split_events

    out_dir = Path(out_dir)
    db_path = out_dir / "warehouse.duckdb"
    batch = Path(DEFAULT_DB_PATH)
    if out_dir.resolve() in {batch.parent.resolve(), batch.resolve().parent}:
        raise ValueError("the load benchmark must not write into the batch warehouse directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    t0 = time.perf_counter()
    events, _ = simulate_events_dataset(num_users=users, days=days, seed=seed, ocr_noise=ocr_noise)
    raw, _ = split_events(events)
    simulate_seconds = round(time.perf_counter() - t0, 2)

    timings: Dict[str, float] = {}
    con = duckdb.connect(str(db_path))
    try:
        migrate(con)
        counts = load_raw_events(con, raw, workers=workers, timings=timings)
    finally:
        con.close()
    return {
        "users": users, "days": days, "seed": seed, "ocr_noise": ocr_noise, "workers": workers,
        "events": int(raw.height), "counts": {k: int(v) for k, v in counts.items()},
        "simulate_seconds": simulate_seconds, **timings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time a default warehouse load (gate G1.3-11).")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--users", type=int, default=400)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--workers", type=int, default=SIMULATION_LOAD_WORKERS)
    parser.add_argument("--no-report", action="store_true", help="print without writing the report")
    args = parser.parse_args()

    result = time_load(Path(args.out), args.users, args.days, workers=args.workers)
    print(result)
    if not args.no_report:
        from src.release.reporting import write_report

        print(f"Wrote {write_report('v1_3_load', '1.3', result)}")

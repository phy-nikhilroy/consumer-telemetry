"""
Calibration of the simulated warehouse against the RetailRocket warehouse.

Both warehouses are sessionized `fact_events` tables, so one SQL function computes every
quantity on either side: events per user per active day, median and 90th percentile events per
session, the single-event session share, the cart-to-view and buy-to-cart ratios, and the share
of users with at least one purchase. The report `reports/v1_4_sim_calibration.json` holds both
columns and `deliberate_differences`, which explains every quantity kept different on purpose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import duckdb  # noqa: E402

from src.config import DATA_DIR, DEFAULT_DB_PATH  # noqa: E402

RETAILROCKET_DB_PATH = DATA_DIR / "external" / "retailrocket.duckdb"
CALIBRATION_REPORT = "v1_4_sim_calibration"
QUANTITIES = (
    "events_per_user_per_active_day", "median_events_per_session", "p90_events_per_session",
    "single_event_session_share", "cart_to_view_ratio", "buy_to_cart_ratio", "share_users_with_purchase",
)

# Every quantity the simulator keeps different from RetailRocket on purpose, and why.
DELIBERATE_DIFFERENCES: List[str] = [
    "share_users_with_purchase is far higher than RetailRocket's: the RetailRocket warehouse is a sample "
    "of site visitors, most of whom never buy, while every simulated user is a returning customer observed "
    "for the whole window. The simulator keeps each binary 7-day target between 5 and 70 percent positive "
    "(tests/test_features.py, gate G1.4-06) so the models have both classes to learn.",
    "buy_to_cart_ratio is higher because a simulated buy may skip the cart (a direct buy from the product "
    "page), while RetailRocket logs a transaction only after an add-to-cart event.",
    "events_per_user_per_active_day is lower: RetailRocket's active days are dominated by a few heavy "
    "visitors, while each simulated shopper starts about one session a day of three to six screens. That "
    "keeps the default 1,000-user, 60-day run near 175,000 screens, which the extraction loader handles in "
    "about ten minutes.",
    "median_events_per_session is higher and single_event_session_share lower: RetailRocket sessions are "
    "mostly one product view from a search result, while each simulated session has an intent family and a "
    "funnel, so the sessionizer and the comparison detector have bursts to recover (gate G1.4-05).",
    "p90_events_per_session is lower: the simulator caps a session at 30 screens and exits the funnel after "
    "a buy, while RetailRocket has long browsing sessions by crawlers and heavy visitors.",
]


def calibration_stats(con: duckdb.DuckDBPyConnection) -> Dict[str, Optional[float]]:
    """The calibration quantities of one sessionized `fact_events` table."""
    per_user = con.execute(
        """
        SELECT avg(n * 1.0 / days) FROM (
            SELECT user_id, count(*) AS n, count(DISTINCT CAST(timestamp AS DATE)) AS days
            FROM fact_events GROUP BY user_id)
        """
    ).fetchone()[0]
    sessions = con.execute(
        """
        SELECT median(n), quantile_cont(n, 0.9), avg(CASE WHEN n = 1 THEN 1.0 ELSE 0.0 END)
        FROM (SELECT session_id, count(*) AS n FROM fact_events
              WHERE session_id IS NOT NULL GROUP BY session_id)
        """
    ).fetchone()
    views, carts, buys, users, buyers = con.execute(
        """
        SELECT count(*) FILTER (WHERE action = 'view'), count(*) FILTER (WHERE action = 'cart'),
               count(*) FILTER (WHERE action = 'buy'), count(DISTINCT user_id),
               count(DISTINCT user_id) FILTER (WHERE action = 'buy')
        FROM fact_events
        """
    ).fetchone()

    def ratio(a: float, b: float) -> Optional[float]:
        return round(a / b, 4) if b else None

    def rnd(v: Optional[float]) -> Optional[float]:
        return None if v is None else round(float(v), 4)

    return {
        "events_per_user_per_active_day": rnd(per_user),
        "median_events_per_session": rnd(sessions[0]),
        "p90_events_per_session": rnd(sessions[1]),
        "single_event_session_share": rnd(sessions[2]),
        "cart_to_view_ratio": ratio(carts, views),
        "buy_to_cart_ratio": ratio(buys, carts),
        "share_users_with_purchase": ratio(buyers, users),
    }


def warehouse_stats(db_path: Union[str, Path]) -> Dict[str, Optional[float]]:
    """`calibration_stats` of the warehouse at `db_path`, opened read only."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return calibration_stats(con)
    finally:
        con.close()


def build_calibration(
    sim_db: Union[str, Path] = DEFAULT_DB_PATH,
    rr_db: Union[str, Path] = RETAILROCKET_DB_PATH,
) -> Dict[str, object]:
    """Both columns, keyed by quantity, plus the deliberate differences."""
    sim, rr = warehouse_stats(sim_db), warehouse_stats(rr_db)
    return {
        "quantities": {q: {"simulated": sim[q], "retailrocket": rr[q]} for q in QUANTITIES},
        "simulated": sim,
        "retailrocket": rr,
        "deliberate_differences": list(DELIBERATE_DIFFERENCES),
    }


def write_calibration_report(results: Dict[str, object], seed: Optional[int] = None) -> Path:
    """Writes `reports/v1_4_sim_calibration.json` with the standard envelope."""
    from src.release.reporting import write_report

    return write_report(CALIBRATION_REPORT, "1.4", results, seed=seed)


if __name__ == "__main__":
    from src.config import RANDOM_SEED

    parser = argparse.ArgumentParser(description="Compare the simulated warehouse with RetailRocket.")
    parser.add_argument("--sim-db", type=str, default=str(DEFAULT_DB_PATH), help="Simulated warehouse")
    parser.add_argument("--rr-db", type=str, default=str(RETAILROCKET_DB_PATH), help="RetailRocket warehouse")
    parser.add_argument("--no-report", action="store_true", help="Print only; do not write the report")
    args = parser.parse_args()
    out = build_calibration(args.sim_db, args.rr_db)
    width = max(len(q) for q in QUANTITIES)
    print(f"{'quantity':<{width}}  simulated  retailrocket")
    for q in QUANTITIES:
        print(f"{q:<{width}}  {out['simulated'][q]!s:>9}  {out['retailrocket'][q]!s:>12}")
    if not args.no_report:
        print(f"wrote {write_calibration_report(out, seed=RANDOM_SEED)}")

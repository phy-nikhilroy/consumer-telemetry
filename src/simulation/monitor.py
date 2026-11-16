"""
Terminal monitor: prints one line of live activity from the API at a fixed interval.

    [12:34:56] Events today: 847 | Users active: 23 | Last purchase: user_42 bought "iPhone 15" on amazon_in

"Today" is the trailing 24 hours of the API's serving clock, which in the default
simulated mode continues the simulated timeline rather than the wall clock.

    .venv/bin/python src/simulation/monitor.py --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional

# Ensure consumer-telemetry root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx

DEFAULT_API_URL = os.environ.get("CT_API_URL", "http://localhost:8000")


def format_activity_line(activity: Dict[str, Any], stamp: Optional[str] = None) -> str:
    """Formats an `/analytics/activity` response as one monitor line."""
    stamp = stamp or datetime.now().strftime("%H:%M:%S")
    purchase = activity.get("last_purchase")
    if purchase:
        title = purchase.get("product_title") or "an unknown product"
        purchase_text = f'{purchase["user_id"]} bought "{title}" on {purchase["app_name"]}'
    else:
        purchase_text = "none yet"
    return (
        f"[{stamp}] Events today: {activity['events_in_window']:,} | "
        f"Users active: {activity['active_users_in_window']:,} | "
        f"Live events: {activity.get('live_events', 0):,} | "
        f"Last purchase: {purchase_text}"
    )


def run_monitor(
    api_url: str = DEFAULT_API_URL,
    interval: float = 5.0,
    iterations: Optional[int] = None,
    client: Optional[httpx.Client] = None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> List[str]:
    """
    Polls the activity endpoint and logs a line each time; returns the lines printed.

    Runs forever when `iterations` is None. Connection errors are reported on the line
    instead of stopping the monitor.
    """
    owns_client = client is None
    http = client or httpx.Client(base_url=api_url, timeout=10.0)
    lines: List[str] = []
    done = 0
    try:
        while iterations is None or done < iterations:
            try:
                response = http.get("/analytics/activity", params={"window_hours": 24})
                response.raise_for_status()
                line = format_activity_line(response.json())
            except httpx.HTTPError as exc:
                line = f"[{datetime.now():%H:%M:%S}] API unavailable at {api_url}: {exc}"
            log(line)
            lines.append(line)
            done += 1
            if iterations is None or done < iterations:
                sleep(interval)
    except KeyboardInterrupt:
        log("Monitor stopped.")
    finally:
        if owns_client:
            http.close()
    return lines


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print live API activity to the terminal.")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="API base URL (env CT_API_URL)")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between updates")
    parser.add_argument("--iterations", type=int, default=0, help="Updates to print; 0 runs until stopped")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_monitor(api_url=args.api_url, interval=args.interval, iterations=args.iterations or None)

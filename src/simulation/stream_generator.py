"""
Live data simulation stream: sends a realistic screen-text event to the API every few seconds.

Each event picks a user (mostly existing users, occasionally a brand-new one), a shopping
app, a catalog product, and an action, renders the noisy screen text with the same
generator the behavioral simulator uses, and posts it to `POST /ingest`. The API stamps
it with its serving clock, extracts the product, and stores it, so the warehouse grows
in real time and every prediction endpoint reflects the new activity.

Run against a local API:
    .venv/bin/python src/simulation/stream_generator.py --api-url http://localhost:8000

Send a fixed number of events quickly (useful for demos and tests):
    .venv/bin/python src/simulation/stream_generator.py --count 20 --min-interval 0.2 --max-interval 0.5

The generator deliberately imports nothing from the NLP stack, so it starts instantly and
fits on a small machine next to the API.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional

# Ensure consumer-telemetry root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx

from src.ingestion.catalog import build_catalog
from src.ingestion.synthetic_data import generate_app_screen_text

DEFAULT_API_URL = os.environ.get("CT_API_URL", "http://localhost:8000")
APPS = ["amazon_in", "flipkart", "walmart", "amazon_us"]
ACTION_WEIGHTS = {"view": 0.72, "cart": 0.18, "buy": 0.10}
DISCOUNTS = [0, 0, 10, 15, 20, 35, 45]
# The v1.2 family catalog (303 SKUs), the same products the behaviour engine simulates.
STREAM_CATALOG: List[dict] = build_catalog()


def fetch_user_pool(client: httpx.Client, limit: int = 1000) -> List[str]:
    """Returns existing user IDs from the API, most recently active first."""
    response = client.get("/users", params={"limit": limit})
    response.raise_for_status()
    return [row["user_id"] for row in response.json()]


def build_event(
    rng: random.Random,
    user_pool: List[str],
    new_user_rate: float = 0.05,
    new_user_prefix: str = "live_user",
) -> Dict[str, Any]:
    """
    Builds one `POST /ingest` payload.

    With probability `new_user_rate` (or always, when the pool is empty) the event
    belongs to a new user, who is added to the pool so later events can return to them.
    """
    if not user_pool or rng.random() < new_user_rate:
        user_id = f"{new_user_prefix}_{rng.randint(10_000, 99_999)}"
        user_pool.append(user_id)
    else:
        user_id = rng.choice(user_pool)

    item = rng.choice(STREAM_CATALOG)
    app = rng.choice(APPS)
    action = rng.choices(list(ACTION_WEIGHTS), weights=list(ACTION_WEIGHTS.values()))[0]
    discount = rng.choice(DISCOUNTS)
    rating = round(rng.uniform(3.5, 4.9), 1)
    text = generate_app_screen_text(item, app, action, discount, rating, rng=rng)
    return {"raw_text": text, "user_id": user_id, "app_name": app}


def describe_response(payload: Dict[str, Any], body: Dict[str, Any]) -> str:
    """Formats one log line for an ingested event."""
    product = body.get("product", {})
    title = product.get("canonical_name") or product.get("product_title") or "unknown product"
    price = product.get("price")
    currency = product.get("currency") or ""
    price_text = f" {currency} {price:,.2f}" if isinstance(price, (int, float)) else ""
    extra = []
    if body.get("is_new_user"):
        extra.append("new user")
    if body.get("cross_app_comparisons_detected"):
        extra.append(f"{body['cross_app_comparisons_detected']} cross-app comparison(s)")
    suffix = f" [{', '.join(extra)}]" if extra else ""
    return (
        f"{payload['user_id']} {body.get('stored_action', '?')} \"{title}\"{price_text} "
        f"on {payload['app_name']} (session {body.get('session_id')}){suffix}"
    )


def run_stream(
    api_url: str = DEFAULT_API_URL,
    min_interval: float = 5.0,
    max_interval: float = 30.0,
    count: Optional[int] = None,
    seed: Optional[int] = None,
    new_user_rate: float = 0.05,
    client: Optional[httpx.Client] = None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> Dict[str, int]:
    """
    Sends events until `count` is reached (forever when None) and returns tallies.

    A failed request is logged and the stream carries on, so a brief API restart does
    not kill a long-running demo stream. `client`, `sleep`, and `log` are injectable so
    the loop can be tested without a network or real waiting.
    """
    if min_interval < 0 or max_interval < min_interval:
        raise ValueError("Intervals must satisfy 0 <= min_interval <= max_interval")

    rng = random.Random(seed)
    owns_client = client is None
    http = client or httpx.Client(base_url=api_url, timeout=30.0)
    tallies = {"sent": 0, "ok": 0, "failed": 0}

    try:
        try:
            user_pool = fetch_user_pool(http)
        except httpx.HTTPError as exc:
            log(f"Could not fetch existing users ({exc}); starting with new users only.")
            user_pool = []
        log(f"Streaming to {api_url} with {len(user_pool)} known users. Press Ctrl+C to stop.")

        while count is None or tallies["sent"] < count:
            payload = build_event(rng, user_pool, new_user_rate=new_user_rate)
            tallies["sent"] += 1
            stamp = datetime.now().strftime("%H:%M:%S")
            try:
                response = http.post("/ingest", json=payload)
                response.raise_for_status()
                tallies["ok"] += 1
                log(f"[{stamp}] {describe_response(payload, response.json())}")
            except httpx.HTTPError as exc:
                tallies["failed"] += 1
                log(f"[{stamp}] ingest failed for {payload['user_id']}: {exc}")

            if count is None or tallies["sent"] < count:
                sleep(rng.uniform(min_interval, max_interval))
    except KeyboardInterrupt:
        log("Stream stopped.")
    finally:
        if owns_client:
            http.close()

    log(f"Sent {tallies['sent']} events: {tallies['ok']} stored, {tallies['failed']} failed.")
    return tallies


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream simulated screen-text events to the live API.")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="API base URL (env CT_API_URL)")
    parser.add_argument("--min-interval", type=float, default=5.0, help="Minimum seconds between events")
    parser.add_argument("--max-interval", type=float, default=30.0, help="Maximum seconds between events")
    parser.add_argument("--count", type=int, default=0, help="Events to send; 0 streams until stopped")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for a reproducible stream")
    parser.add_argument("--new-user-rate", type=float, default=0.05, help="Share of events from brand-new users")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_stream(
        api_url=args.api_url,
        min_interval=args.min_interval,
        max_interval=args.max_interval,
        count=args.count or None,
        seed=args.seed,
        new_user_rate=args.new_user_rate,
    )

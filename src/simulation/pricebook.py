"""Deterministic price book: per (sku, app) price factors and promotion episodes.

A quote depends only on the seed, the sku, the app and the day, never on call order.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import USD_TO_INR_RATE  # noqa: E402

INDIAN_APPS: frozenset[str] = frozenset({"amazon_in", "flipkart"})
PROMO_DISCOUNTS: tuple[int, ...] = (10, 15, 20, 30, 40)
PROMO_START_PROB: float = 1 / 20
PROMO_MIN_DAYS: int = 2
PROMO_MAX_DAYS: int = 5
FACTOR_SIGMA: float = 0.06


@dataclass(frozen=True)
class Quote:
    """A price as a screen shows it, with its INR equivalent."""

    price: float
    mrp: float
    discount_pct: int
    currency: str
    price_inr: float


def _rng(seed: int, sku_id: str, app: str) -> np.random.Generator:
    """Generator keyed by (seed, sku, app) through a stable hash."""
    digest = hashlib.sha256(f"{seed}|{sku_id}|{app}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


class PriceBook:
    """Quotes for every (sku, app, day) of a simulated period."""

    def __init__(self, catalog: Sequence[dict], apps: Sequence[str], days: int, seed: int) -> None:
        self._base = {str(it["sku_id"]): float(it["base_inr"]) for it in catalog}
        self._apps = tuple(apps)
        self._days = int(days)
        self._seed = int(seed)
        self._state: dict[tuple[str, str], tuple[float, np.random.Generator, list[int]]] = {}

    def _discount(self, sku_id: str, app: str, day: int) -> tuple[float, int]:
        """Return (factor, discount) for a day, extending the schedule lazily.

        The draw sequence per (sku, app) never depends on the horizon, so quotes are
        prefix-stable in ``days`` and defined beyond it.
        """
        key = (sku_id, app)
        if key not in self._state:
            rng = _rng(self._seed, sku_id, app)
            self._state[key] = (float(rng.lognormal(0.0, FACTOR_SIGMA)), rng, [])
        factor, rng, disc = self._state[key]
        while len(disc) <= day:
            if rng.random() < PROMO_START_PROB:
                length = int(rng.integers(PROMO_MIN_DAYS, PROMO_MAX_DAYS + 1))
                d = int(PROMO_DISCOUNTS[int(rng.integers(0, len(PROMO_DISCOUNTS)))])
                disc.extend([d] * length)
            else:
                disc.append(0)
        return factor, disc[day]

    def quote(self, sku_id: str, app: str, day: int) -> Quote:
        """Return the quote for a sku on an app on a day (0-based; any day >= 0)."""
        if sku_id not in self._base:
            raise ValueError(f"unknown sku {sku_id}")
        if app not in self._apps:
            raise ValueError(f"unknown app {app}")
        if day < 0:
            raise ValueError(f"negative day {day}")
        factor, d = self._discount(sku_id, app, day)
        mrp_inr = self._base[sku_id] * factor
        price_inr_raw = mrp_inr * (1 - d / 100)
        if app in INDIAN_APPS:
            mrp, price = float(round(mrp_inr)), float(round(price_inr_raw))
            return Quote(price, mrp, d, "INR", price)
        mrp = round(mrp_inr / USD_TO_INR_RATE, 2)
        price = round(price_inr_raw / USD_TO_INR_RATE, 2)
        return Quote(price, mrp, d, "USD", price * USD_TO_INR_RATE)


if __name__ == "__main__":
    from src.ingestion.catalog import build_catalog

    ap = argparse.ArgumentParser(description="Print price book quotes for one SKU.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--sku-index", type=int, default=0)
    a = ap.parse_args()
    cat = build_catalog()
    apps = ["amazon_in", "flipkart", "walmart", "amazon_us"]
    pb = PriceBook(cat, apps, a.days, a.seed)
    sku = cat[a.sku_index]["sku_id"]
    for d in range(a.days):
        print(d, [pb.quote(sku, ap_, d) for ap_ in apps[:1]])

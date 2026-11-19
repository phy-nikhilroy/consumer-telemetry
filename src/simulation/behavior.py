"""
Behaviour engine: simulated shoppers who act for reasons, with the reasons kept as ground truth.

Each shopper has persistent traits (archetype, activity, purchase propensity, price sensitivity,
app loyalty, price tier, category preference). Each day the shopper may drift to another
archetype (weekly), churn for good (from `SIMULATION_CHURN_START_DAY`), and otherwise starts a
Poisson number of sessions around two daily peaks. A session has one intent family and walks a
funnel state machine (browse, consider, compare, cart, buy, exit). In the compare state the
shopper views the same SKU on another app one to ten minutes later; after a comparison the
purchase app is drawn from a multinomial logit in which the price term is scaled by
`price_beta`. Prices come from the price book, so a comparison sees real price differences.

Every event is rendered as a raw screen (focal listing plus 0 to 3 context listings). The raw
frame is all the warehouse ever sees; the truth frame (archetype, true session, intent, funnel
state, SKU, app, comparison flag, cheapest-seen flag) is read only by evaluation code.

Randomness: the engine owns `random.Random(seed)` for profiles. Each (user, day) draws from its
own `random.Random` seeded by a stable hash of (seed, user, day), and rendering uses a second
one, so a day's behaviour depends only on the user's state at the start of that day. That is
what lets `continue_user` reproduce `simulate` exactly, and lets two OCR noise levels share one
behavioural stream. The module-level `random` functions are never called.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import polars as pl  # noqa: E402

from src.config import (  # noqa: E402
    ARCHETYPE_DRIFT_WEEKLY,
    PRICE_BETA,
    PRICE_TIERS,
    SIMULATION_APPS,
    SIMULATION_CHURN_START_DAY,
    SIMULATION_MAX_CONTEXT_LISTINGS,
    SIMULATION_OCR_NOISE,
    SIMULATION_VIEW_MEMORY_DAYS,
)
from src.ingestion.screen_generator import apply_ocr_noise, render_listing, render_screen  # noqa: E402

APPS: Tuple[str, ...] = tuple(SIMULATION_APPS)
ARCHETYPES: Tuple[str, ...] = (
    "loyalist", "comparison_shopper", "bargain_hunter", "impulse_buyer", "window_shopper",
)
RAW_COLUMNS: Tuple[str, ...] = ("event_id", "user_id", "timestamp", "app_name", "raw_text")
# The spec's truth columns, then the v1.3 columns `noise_propagation` scores extraction with.
TRUTH_COLUMNS: Tuple[str, ...] = (
    "event_id", "user_id", "archetype", "true_session_id", "intent_family", "state",
    "true_sku_id", "true_action", "true_price_inr", "true_app", "is_deliberate_comparison",
    "chose_cheapest_seen", "true_sku_key", "true_price", "true_currency",
)
STATE_ACTION = {"browse": "view", "consider": "view", "compare": "view", "cart": "cart", "buy": "buy"}

# Per-archetype rates. `sessions` is the mean sessions per day before the user's activity and
# the weekend factor. `browse` and `consider` give the next-state probabilities out of browse
# and consider; `compare` is the probability out of consider of comparing on another app,
# `cart` and `buy` the base probabilities out of consider (compare multiplies them by
# COMPARE_CONVERSION_LIFT), `cart_buy` the probability of buying out of cart. `sensitivity`
# scales the price term and the discount uplift; `loyalty_u` is the logit weight of the
# primary app; `start_primary` the probability a session starts on it; `hazard` the daily
# churn probability; `tiers` the weights of budget, mid and premium price tiers.
ARCHETYPE_BEHAVIOUR: Dict[str, Dict[str, Any]] = {
    "loyalist": dict(sessions=0.75, browse=(0.50, 0.30), consider=0.15, compare=0.04, cart=0.18,
                     buy=0.07, cart_buy=0.55, sensitivity=0.5, loyalty_u=2.0, start_primary=0.85,
                     hazard=0.0035, tiers=(0.20, 0.40, 0.40)),
    "comparison_shopper": dict(sessions=0.95, browse=(0.50, 0.30), consider=0.15, compare=0.40,
                               cart=0.08, buy=0.02, cart_buy=0.45, sensitivity=1.6, loyalty_u=0.6,
                               start_primary=0.45, hazard=0.0050, tiers=(0.34, 0.40, 0.26)),
    "bargain_hunter": dict(sessions=0.85, browse=(0.45, 0.35), consider=0.15, compare=0.30,
                           cart=0.10, buy=0.03, cart_buy=0.40, sensitivity=2.0, loyalty_u=0.4,
                           start_primary=0.40, hazard=0.0055, tiers=(0.60, 0.30, 0.10)),
    "impulse_buyer": dict(sessions=0.80, browse=(0.60, 0.20), consider=0.10, compare=0.05,
                          cart=0.20, buy=0.12, cart_buy=0.60, sensitivity=0.7, loyalty_u=1.0,
                          start_primary=0.60, hazard=0.0070, tiers=(0.34, 0.40, 0.26)),
    "window_shopper": dict(sessions=1.10, browse=(0.35, 0.45), consider=0.20, compare=0.10,
                           cart=0.04, buy=0.005, cart_buy=0.15, sensitivity=0.8, loyalty_u=0.8,
                           start_primary=0.55, hazard=0.0080, tiers=(0.25, 0.35, 0.40)),
}

WEEKEND_FACTOR = 1.3
# Session start hours: a mixture of two normal peaks (mean, sd, weight), clipped to the day.
SESSION_PEAKS: Tuple[Tuple[float, float, float], ...] = ((13.0, 1.6, 0.5), (21.0, 1.1, 0.5))
SESSION_EARLIEST_HOUR = 8.5
SESSION_LATEST_HOUR = 23.25
MIN_SESSION_GAP_MINUTES = 31
MAX_SESSION_EVENTS = 30
EVENT_GAP_SECONDS = (20, 180)
COMPARE_GAP_SECONDS = (60, 600)
# Probability out of compare of comparing on yet another app, and the conversion lift after it.
COMPARE_AGAIN = 0.30
COMPARE_CONVERSION_LIFT = 1.6
# Out of cart: probability of going back to consider the target.
CART_RECONSIDER = 0.15
# Share of exit probability kept on the first event of a session (sessions are bursty).
FIRST_EVENT_EXIT_SHARE = 0.3
# How strongly a discount lifts cart and buy: factor 1 + DISCOUNT_UPLIFT * sensitivity * pct / 100.
DISCOUNT_UPLIFT = 2.0
# Probability that a session returns to a SKU viewed within the view memory.
RETURN_TO_VIEWED = 0.25
TIER_AFFINITY = 0.78
SAME_FAMILY_CONTEXT = 0.5
SALT_DAY = "day"
SALT_RENDER = "render"


def _stable_seed(*parts: Any) -> int:
    """A 64-bit seed from a stable hash of the parts (Python's hash() is salted per process)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def sku_key(item: dict) -> str:
    """The v1.3 ground-truth identity of a catalog item: brand, title, and specs."""
    return " ".join([item["brand"], item["title"], *item.get("specs", [])]).strip()


def _poisson(rng: random.Random, lam: float) -> int:
    """A Poisson draw by inversion (random.Random has no Poisson sampler)."""
    if lam <= 0:
        return 0
    threshold, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= threshold:
            return k
        k += 1


@dataclass
class UserProfile:
    """Persistent shopper traits, drawn once by `BehaviorEngine.create_profiles`."""

    user_id: str
    archetype: str
    primary_app: str
    activity_level: float
    buy_propensity: float
    price_sensitivity: float
    loyalty_jitter: float
    price_tier: str
    category_weights: Dict[str, float]


@dataclass
class UserState:
    """
    Everything a day of simulation reads besides the profile. `simulate` leaves each user's end
    state in `engine.states`; `continue_user` picks up from it.
    """

    origin: datetime
    archetype: str
    next_day: int = 0
    churned_day: Optional[int] = None
    event_seq: int = 0
    session_seq: int = 0
    # SKU id to the last day index it was viewed, pruned to the view memory.
    recent_views: Dict[str, int] = field(default_factory=dict)


class BehaviorEngine:
    """Simulates shoppers over a catalog and a price book; see the module docstring."""

    def __init__(
        self,
        catalog: Sequence[dict],
        pricebook: Any,
        seed: int,
        price_beta: float = PRICE_BETA,
        ocr_noise: float = SIMULATION_OCR_NOISE,
    ) -> None:
        self.catalog = list(catalog)
        self.pricebook = pricebook
        self.seed = int(seed)
        self.price_beta = float(price_beta)
        self.ocr_noise = float(ocr_noise)
        self._rng = random.Random(self.seed)
        self.states: Dict[str, UserState] = {}

        self._by_sku = {item["sku_id"]: item for item in self.catalog}
        self._families: Dict[str, List[dict]] = {}
        for item in self.catalog:
            self._families.setdefault(item["family_id"], []).append(item)
        self._categories = sorted({item["category"] for item in self.catalog})
        # Families per (category, tier); a family's tier is its first (cheapest) variant's.
        self._family_ids_by_cat: Dict[str, List[str]] = {c: [] for c in self._categories}
        self._family_ids_by_cat_tier: Dict[Tuple[str, str], List[str]] = {}
        for fam_id, items in self._families.items():
            cat, tier = items[0]["category"], items[0].get("tier") or self._tier_of(items[0])
            self._family_ids_by_cat[cat].append(fam_id)
            self._family_ids_by_cat_tier.setdefault((cat, tier), []).append(fam_id)
        self._skus_by_cat: Dict[str, List[dict]] = {c: [] for c in self._categories}
        for item in self.catalog:
            self._skus_by_cat[item["category"]].append(item)
        self._rating = {
            sku: round(3.5 + random.Random(_stable_seed(self.seed, "rating", sku)).randint(0, 14) / 10, 1)
            for sku in self._by_sku
        }

    @staticmethod
    def _tier_of(item: dict) -> str:
        """The price tier of an item's INR base price."""
        price = float(item["base_inr"])
        for name, (low, high) in PRICE_TIERS.items():
            if low <= price < high:
                return name
        return list(PRICE_TIERS)[-1]

    # ------------------------------------------------------------------ profiles

    def create_profiles(self, num_users: int) -> List[UserProfile]:
        """`num_users` profiles, archetypes in rotation, traits from the engine's generator."""
        rng = self._rng
        rng.seed(self.seed)
        profiles: List[UserProfile] = []
        width = max(4, len(str(num_users)))
        for i in range(1, num_users + 1):
            archetype = ARCHETYPES[(i - 1) % len(ARCHETYPES)]
            params = ARCHETYPE_BEHAVIOUR[archetype]
            primary_app = rng.choice(APPS)
            activity = min(2.2, max(0.45, rng.lognormvariate(0.0, 0.32)))
            propensity = min(2.6, max(0.25, rng.lognormvariate(0.0, 0.45)))
            sensitivity = min(2.5, max(0.40, rng.lognormvariate(0.0, 0.35)))
            loyalty_jitter = min(1.6, max(0.6, rng.lognormvariate(0.0, 0.2)))
            tier = rng.choices(list(PRICE_TIERS), weights=params["tiers"])[0]
            raw = {c: rng.gammavariate(1.0, 1.0) for c in self._categories}
            favourite = rng.choice(self._categories)
            raw[favourite] += 2.0
            total = sum(raw.values())
            profiles.append(UserProfile(
                user_id=f"user_{i:0{width}d}", archetype=archetype, primary_app=primary_app,
                activity_level=activity, buy_propensity=propensity, price_sensitivity=sensitivity,
                loyalty_jitter=loyalty_jitter, price_tier=tier,
                category_weights={c: v / total for c, v in raw.items()},
            ))
        return profiles

    # ------------------------------------------------------------------ public runs

    def simulate(
        self, profiles: Sequence[UserProfile], start: datetime, days: int
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Simulates `days` days from `start` for every profile. Returns (raw_events, truth), both
        ordered by (timestamp, event_id) with the same event ids. End states go to `self.states`.
        """
        return events_to_frames(self.simulate_events(profiles, start, days))

    def simulate_events(
        self, profiles: Sequence[UserProfile], start: datetime, days: int
    ) -> List[Dict[str, Any]]:
        """`simulate` as event dicts (raw and truth columns) in (timestamp, event_id) order."""
        events: List[Dict[str, Any]] = []
        for profile in profiles:
            state = UserState(origin=start, archetype=profile.archetype)
            events.extend(self._run_days(profile, state, 0, days))
            self.states[profile.user_id] = state
        events.sort(key=lambda e: (e["timestamp"], e["event_id"]))
        return events

    def continue_user(
        self, profile: UserProfile, state: UserState, now: datetime, until: datetime
    ) -> List[Dict[str, Any]]:
        """
        Continues one user from `state` and returns the events in [now, until), each a dict of
        the raw and truth columns, in (timestamp, event_id) order; `state` becomes the end
        state. Whole days are simulated from `state.next_day`, so continuing from a saved state
        gives exactly the events `simulate` would have generated for that period.
        """
        if until <= now:
            return []
        last_day = math.ceil((until - state.origin) / timedelta(days=1))
        out = self._run_days(profile, state, state.next_day, last_day)
        return [e for e in out if now <= e["timestamp"] < until]

    # ------------------------------------------------------------------ days

    def _run_days(self, profile: UserProfile, state: UserState, first: int, last: int) -> List[Dict[str, Any]]:
        """Simulates day indices [first, last) in order, updating `state`."""
        out: List[Dict[str, Any]] = []
        for day in range(max(first, state.next_day), last):
            out.extend(self._run_day(profile, state, day))
            state.next_day = day + 1
        return out

    def _run_day(self, profile: UserProfile, state: UserState, day: int) -> List[Dict[str, Any]]:
        """One day: drift, churn, then the day's sessions."""
        if state.churned_day is not None:
            return []
        rng = random.Random(_stable_seed(self.seed, SALT_DAY, profile.user_id, day))
        render_rng = random.Random(_stable_seed(self.seed, SALT_RENDER, profile.user_id, day))

        if day > 0 and day % 7 == 0 and rng.random() < ARCHETYPE_DRIFT_WEEKLY:
            state.archetype = rng.choice([a for a in ARCHETYPES if a != state.archetype])
        params = ARCHETYPE_BEHAVIOUR[state.archetype]
        if day >= SIMULATION_CHURN_START_DAY and rng.random() < params["hazard"]:
            state.churned_day = day
            return []

        state.recent_views = {
            sku: d for sku, d in state.recent_views.items() if day - d <= SIMULATION_VIEW_MEMORY_DAYS
        }
        day_start = state.origin + timedelta(days=day)
        midnight = datetime.combine(day_start.date() + timedelta(days=1), datetime.min.time())
        weekend = WEEKEND_FACTOR if day_start.weekday() >= 5 else 1.0
        n_sessions = _poisson(rng, params["sessions"] * profile.activity_level * weekend)
        starts = sorted(self._session_start_hour(rng) for _ in range(n_sessions))

        out: List[Dict[str, Any]] = []
        latest_start = datetime.combine(day_start.date(), datetime.min.time()) + timedelta(hours=SESSION_LATEST_HOUR)
        prev_end: Optional[datetime] = None
        for hour in starts:
            t0 = datetime.combine(day_start.date(), datetime.min.time()) + timedelta(
                seconds=int(round(hour * 3600)))
            if t0 < day_start:
                t0 = day_start
            if prev_end is not None and t0 < prev_end + timedelta(minutes=MIN_SESSION_GAP_MINUTES):
                t0 = prev_end + timedelta(minutes=MIN_SESSION_GAP_MINUTES, seconds=rng.randint(0, 1800))
            if t0 > latest_start:
                break
            session = self._run_session(profile, state, day, t0, midnight, rng, render_rng)
            if session:
                out.extend(session)
                prev_end = session[-1]["timestamp"]
        return out

    @staticmethod
    def _session_start_hour(rng: random.Random) -> float:
        """A session start hour from the two-peak mixture, clipped to the active day."""
        u, acc = rng.random(), 0.0
        mean, sd = SESSION_PEAKS[-1][0], SESSION_PEAKS[-1][1]
        for m, s, w in SESSION_PEAKS:
            acc += w
            if u < acc:
                mean, sd = m, s
                break
        return min(SESSION_LATEST_HOUR, max(SESSION_EARLIEST_HOUR, rng.gauss(mean, sd)))

    # ------------------------------------------------------------------ sessions

    def _pick_family(self, profile: UserProfile, rng: random.Random) -> str:
        """The session's intent family, from category preference and price tier."""
        cats = self._categories
        cat = rng.choices(cats, weights=[profile.category_weights.get(c, 0.0) for c in cats])[0]
        in_tier = self._family_ids_by_cat_tier.get((cat, profile.price_tier), [])
        if in_tier and rng.random() < TIER_AFFINITY:
            return rng.choice(in_tier)
        return rng.choice(self._family_ids_by_cat[cat])

    def _uplift(self, profile: UserProfile, params: Dict[str, Any], discount: int) -> float:
        """Cart and buy multiplier from propensity and the quoted discount."""
        sens = params["sensitivity"] * profile.price_sensitivity
        return profile.buy_propensity * (1.0 + DISCOUNT_UPLIFT * sens * discount / 100.0)

    @staticmethod
    def _draw(rng: random.Random, options: List[Tuple[str, float]], first_event: bool) -> str:
        """Draws the next state; the remainder of the probability mass is exit."""
        total = sum(p for _, p in options)
        if total > 0.95:
            options = [(s, p * 0.95 / total) for s, p in options]
            total = 0.95
        exit_p = 1.0 - total
        if first_event:
            exit_p *= FIRST_EVENT_EXIT_SHARE
        u = rng.random() * (total + exit_p)
        acc = 0.0
        for s, p in options:
            acc += p
            if u < acc:
                return s
        return "exit"

    def _purchase_app(self, profile: UserProfile, params: Dict[str, Any], seen: Dict[str, float],
                      rng: random.Random) -> str:
        """Multinomial logit over the apps seen for the target SKU."""
        loyalty_u = params["loyalty_u"] * profile.loyalty_jitter
        sens = params["sensitivity"] * profile.price_sensitivity
        apps = list(seen)
        utils = [loyalty_u * (a == profile.primary_app) - self.price_beta * sens * math.log(seen[a])
                 for a in apps]
        top = max(utils)
        weights = [math.exp(u - top) for u in utils]
        return rng.choices(apps, weights=weights)[0]

    def _run_session(self, profile: UserProfile, state: UserState, day: int, t0: datetime,
                     midnight: datetime, rng: random.Random, render_rng: random.Random) -> List[Dict[str, Any]]:
        """One session's events through the funnel state machine."""
        params = ARCHETYPE_BEHAVIOUR[state.archetype]
        recent = sorted(state.recent_views)
        if recent and rng.random() < RETURN_TO_VIEWED:
            target = self._by_sku[rng.choice(recent)]
            family = target["family_id"]
            phase = "consider"
        else:
            family = self._pick_family(profile, rng)
            target = rng.choice(self._families[family])
            phase = "browse" if rng.random() < 0.6 else "consider"
        app = profile.primary_app if rng.random() < params["start_primary"] else rng.choice(
            [a for a in APPS if a != profile.primary_app])

        state.session_seq += 1
        session_id = f"{profile.user_id}_s{state.session_seq:05d}"
        seen: Dict[str, float] = {}
        compared = False
        purchase_app: Optional[str] = None
        events: List[Dict[str, Any]] = []
        t = t0
        while phase != "exit" and len(events) < MAX_SESSION_EVENTS:
            if events:
                lo, hi = COMPARE_GAP_SECONDS if phase == "compare" else EVENT_GAP_SECONDS
                t = t + timedelta(seconds=rng.randint(lo, hi))
                if t >= midnight:
                    break
            if phase == "browse":
                others = [s for s in self._families[family] if s["sku_id"] != target["sku_id"]]
                item = rng.choice(others) if others else target
            else:
                item = target
            if phase == "compare":
                app = rng.choice([a for a in APPS if a not in seen])
            elif phase in ("cart", "buy") and compared:
                if purchase_app is None:
                    purchase_app = self._purchase_app(profile, params, seen, rng)
                app = purchase_app

            quote = self.pricebook.quote(item["sku_id"], app, day)
            chose_cheapest = None
            if phase == "buy" and compared:
                chose_cheapest = seen[app] <= min(seen.values())
            events.append(self._event(profile, state, session_id, family, phase, item, app, quote,
                                      t, day, chose_cheapest, render_rng))
            if item is target:
                seen.setdefault(app, quote.price_inr)
            if STATE_ACTION[phase] == "view":
                state.recent_views[item["sku_id"]] = day
            if phase == "compare":
                compared = True
            phase = self._next_phase(phase, profile, params, quote.discount_pct, seen, rng, len(events) == 1)
        return events

    def _next_phase(self, phase: str, profile: UserProfile, params: Dict[str, Any], discount: int,
                    seen: Dict[str, float], rng: random.Random, first_event: bool) -> str:
        """The funnel transition out of `phase`."""
        if phase == "buy":
            return "exit"
        lift = self._uplift(profile, params, discount)
        can_compare = len(seen) < len(APPS) and bool(seen)
        if phase == "browse":
            to_consider, stay = params["browse"]
            return self._draw(rng, [("consider", to_consider), ("browse", stay)], first_event)
        if phase == "consider":
            options = [("consider", params["consider"]), ("browse", 0.12),
                       ("cart", params["cart"] * lift), ("buy", params["buy"] * lift)]
            if can_compare:
                options.append(("compare", params["compare"]))
            return self._draw(rng, options, first_event)
        if phase == "compare":
            options = [("consider", 0.10),
                       ("cart", params["cart"] * lift * COMPARE_CONVERSION_LIFT),
                       ("buy", params["buy"] * lift * COMPARE_CONVERSION_LIFT)]
            if can_compare:
                options.append(("compare", COMPARE_AGAIN))
            return self._draw(rng, options, first_event)
        # cart
        return self._draw(rng, [("buy", min(0.9, params["cart_buy"] * profile.buy_propensity)),
                                ("consider", CART_RECONSIDER)], first_event)

    # ------------------------------------------------------------------ rendering

    def _event(self, profile: UserProfile, state: UserState, session_id: str, family: str, phase: str,
               item: dict, app: str, quote: Any, t: datetime, day: int, chose_cheapest: Optional[bool],
               render_rng: random.Random) -> Dict[str, Any]:
        """One event dict: raw columns, then truth columns."""
        state.event_seq += 1
        action = STATE_ACTION[phase]
        return {
            "event_id": f"{profile.user_id}_e{state.event_seq:06d}",
            "user_id": profile.user_id,
            "timestamp": t,
            "app_name": app,
            "raw_text": self._render(item, app, action, quote, day, render_rng),
            "archetype": state.archetype,
            "true_session_id": session_id,
            "intent_family": family,
            "state": phase,
            "true_sku_id": item["sku_id"],
            "true_action": action,
            "true_price_inr": float(quote.price_inr),
            "true_app": app,
            "is_deliberate_comparison": phase == "compare",
            "chose_cheapest_seen": chose_cheapest,
            "true_sku_key": sku_key(item),
            "true_price": float(quote.price),
            "true_currency": quote.currency,
        }

    def _render(self, item: dict, app: str, action: str, quote: Any, day: int, rng: random.Random) -> str:
        """The screen: focal listing first, then 0 to 3 context listings quoted for the same day."""
        listings = [render_listing(item, app, action, quote.discount_pct, self._rating[item["sku_id"]], rng,
                                   price=quote.price, title_variant=False)]
        count = rng.randint(0, SIMULATION_MAX_CONTEXT_LISTINGS)
        if count:
            family = [s for s in self._families[item["family_id"]] if s["sku_id"] != item["sku_id"]]
            category = [s for s in self._skus_by_cat[item["category"]] if s["sku_id"] != item["sku_id"]]
            pool = family if family and rng.random() < SAME_FAMILY_CONTEXT else category
            for other in rng.sample(pool, min(count, len(pool))):
                q = self.pricebook.quote(other["sku_id"], app, day)
                listings.append(render_listing(other, app, "view", q.discount_pct, self._rating[other["sku_id"]],
                                               rng, price=q.price, title_variant=False))
        return apply_ocr_noise(render_screen(listings, rng), self.ocr_noise, rng)


def events_to_frames(events: Sequence[Dict[str, Any]]) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Splits event dicts into (raw_events, truth), both ordered by (timestamp, event_id)."""
    ordered = sorted(events, key=lambda e: (e["timestamp"], e["event_id"]))
    schema = {
        "event_id": pl.Utf8, "user_id": pl.Utf8, "timestamp": pl.Datetime("us"), "app_name": pl.Utf8,
        "raw_text": pl.Utf8, "archetype": pl.Utf8, "true_session_id": pl.Utf8, "intent_family": pl.Utf8,
        "state": pl.Utf8, "true_sku_id": pl.Utf8, "true_action": pl.Utf8, "true_price_inr": pl.Float64,
        "true_app": pl.Utf8, "is_deliberate_comparison": pl.Boolean, "chose_cheapest_seen": pl.Boolean,
        "true_sku_key": pl.Utf8, "true_price": pl.Float64, "true_currency": pl.Utf8,
    }
    frame = pl.DataFrame(
        {c: [e[c] for e in ordered] for c in schema}, schema=schema
    )
    return frame.select(RAW_COLUMNS), frame.select(TRUTH_COLUMNS)


_PROFILE_FIELDS = (
    "user_id", "archetype", "primary_app", "activity_level", "buy_propensity",
    "price_sensitivity", "loyalty_jitter", "price_tier",
)


def profiles_frame(profiles: Sequence[UserProfile], states: Dict[str, UserState]) -> pl.DataFrame:
    """
    One row per profile with every `UserProfile` field and every `UserState` field. The profile's
    `archetype` keeps its name; the state's is `state_archetype` (it differs after drift). The two
    dict fields are JSON strings. Read back with `read_profiles`.
    """
    import json

    rows: List[Dict[str, Any]] = []
    for p in profiles:
        s = states[p.user_id]
        row = {name: getattr(p, name) for name in _PROFILE_FIELDS}
        row["category_weights"] = json.dumps(p.category_weights)
        row.update(
            origin=s.origin, state_archetype=s.archetype, next_day=s.next_day,
            churned_day=s.churned_day, event_seq=s.event_seq, session_seq=s.session_seq,
            recent_views=json.dumps(s.recent_views),
        )
        rows.append(row)
    schema = {
        "user_id": pl.Utf8, "archetype": pl.Utf8, "primary_app": pl.Utf8,
        "activity_level": pl.Float64, "buy_propensity": pl.Float64,
        "price_sensitivity": pl.Float64, "loyalty_jitter": pl.Float64, "price_tier": pl.Utf8,
        "category_weights": pl.Utf8, "origin": pl.Datetime("us"), "state_archetype": pl.Utf8,
        "next_day": pl.Int64, "churned_day": pl.Int64, "event_seq": pl.Int64,
        "session_seq": pl.Int64, "recent_views": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


def read_profiles(path: Any) -> Tuple[List[UserProfile], Dict[str, UserState]]:
    """Reads a `profiles_frame` parquet back into profiles and per-user end states."""
    import json

    profiles: List[UserProfile] = []
    states: Dict[str, UserState] = {}
    for r in pl.read_parquet(path).iter_rows(named=True):
        profiles.append(UserProfile(
            **{name: r[name] for name in _PROFILE_FIELDS},
            category_weights=json.loads(r["category_weights"]),
        ))
        states[r["user_id"]] = UserState(
            origin=r["origin"], archetype=r["state_archetype"], next_day=r["next_day"],
            churned_day=r["churned_day"], event_seq=r["event_seq"], session_seq=r["session_seq"],
            recent_views=json.loads(r["recent_views"]),
        )
    return profiles, states

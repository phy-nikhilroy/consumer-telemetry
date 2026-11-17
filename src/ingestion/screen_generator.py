"""
Multi-listing screen generator.

Renders catalog items as listings of two to four lines each, joins several listings into one
screen, and applies OCR-style character noise. A v1.0 screen dump held exactly one product on
one line; real result pages and comparison screens hold several, and the extractors must be
measured on that. Every listing ends on its action or stock line, which is the boundary the
segmenter in `src/ingestion/segmenter.py` looks for.

All randomness comes from the `random.Random` passed in, so the same seed renders the same
screen.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Sequence

from src.ingestion.synthetic_data import ACTION_SNIPPETS, DELIVERY_BADGES, _format_inr

# Character confusions a phone OCR pass makes most often, in both directions.
OCR_CONFUSIONS: Dict[str, str] = {
    "O": "0", "0": "O", "l": "1", "1": "l", "S": "5", "5": "S", "B": "8", "8": "B",
}

# Stock lines close a listing whose action is neither buy nor cart.
STOCK_PHRASES: Dict[str, List[str]] = {
    "amazon_in": ["In stock"],
    "amazon_us": ["In Stock"],
    "flipkart": ["In stock"],
    "walmart": ["In stock"],
}

# Actions whose snippet is itself the closing line of a listing.
TERMINAL_ACTIONS = ("buy", "cart")

INR_APPS = ("amazon_in", "flipkart")

CAPACITY_RE = re.compile(r"\b(\d+)\s?(GB|TB)\b", re.IGNORECASE)


def _toggle_capacity_spacing(token: str) -> str:
    """Switches `256GB` to `256 GB` and back."""
    def swap(match: re.Match) -> str:
        sep = "" if match.group(0)[len(match.group(1))] == " " else " "
        return f"{match.group(1)}{sep}{match.group(2)}"
    return CAPACITY_RE.sub(swap, token)


def _render_specs(specs: Sequence[str], rng: random.Random, title_variant: bool) -> str:
    """Renders spec tokens, varying the wording when `title_variant` is set."""
    tokens = [str(s) for s in specs if str(s).strip()]
    if not title_variant:
        return " ".join(tokens)
    if len(tokens) > 1 and rng.random() < 0.5:
        tokens.pop(rng.randrange(len(tokens)))
    rng.shuffle(tokens)
    if rng.random() < 0.5:
        tokens = [_toggle_capacity_spacing(t) for t in tokens]
    return " ".join(tokens)


def _format_price(value: float, currency: str) -> str:
    """Formats a price the way the app of that currency shows it."""
    if currency == "INR":
        return f"₹{_format_inr(int(round(value)))}"
    return f"${value:,.2f}"


def _rating_text(rating: float, rng: random.Random) -> str:
    """One of the v1.0 rating styles, without the unicode stars that hide the value."""
    num_ratings = rng.choice([2847, 123456, 1420, 52100, 890, 43981])
    style = rng.choice(["star_symbol", "text_rating"])
    if style == "star_symbol":
        return f"{rating:.1f} ★ ({num_ratings:,})"
    return f"{rating:.1f} out of 5 stars"


def render_listing(
    item: dict,
    app: str,
    action: str,
    discount_pct: int,
    rating: float,
    rng: random.Random,
    price: Optional[float] = None,
    title_variant: bool = True,
) -> List[str]:
    """
    Renders one catalog item as the lines of one listing.

    `item` carries the catalog keys `brand`, `title`, `specs`, `base_inr`, and `base_usd`.
    The listing has two to four lines. The last line is always the action snippet when the
    action is buy or cart, and a stock line otherwise, so a segmenter can split on it. Other
    action snippets (view, search, wishlist) sit on an earlier line. When `price` is None it
    is drawn within 10 percent of the catalog base price in the app's currency. The brand and
    the title are never altered; `title_variant` varies only the spec tokens.
    """
    if app not in ACTION_SNIPPETS:
        raise ValueError(f"unknown app {app!r}")
    if action not in ACTION_SNIPPETS[app]:
        raise ValueError(f"unknown action {action!r}")

    currency = "INR" if app in INR_APPS else "USD"
    if price is None:
        base = float(item["base_inr"] if currency == "INR" else item["base_usd"])
        price = base * rng.uniform(0.90, 1.10)
        price = float(int(round(price))) if currency == "INR" else round(price, 2)

    head = " ".join(
        part for part in (item["brand"], item["title"], _render_specs(item.get("specs", []), rng, title_variant))
        if part
    )
    price_line = _format_price(price, currency)
    if discount_pct and discount_pct > 0:
        mrp = price / (1 - discount_pct / 100)
        price_line += f" M.R.P: {_format_price(mrp, currency)} ({discount_pct}% off)"

    meta_parts = [_rating_text(rating, rng)] if rating is not None else []
    if action not in TERMINAL_ACTIONS:
        meta_parts.append(rng.choice(ACTION_SNIPPETS[app][action]))
        end_line = rng.choice(STOCK_PHRASES[app])
    else:
        end_line = rng.choice(ACTION_SNIPPETS[app][action])

    n_lines = rng.randint(2, 4)
    if n_lines == 2:
        body = [" ".join([head, price_line] + meta_parts)]
    elif n_lines == 3:
        body = [head, " ".join([price_line] + meta_parts)]
    else:
        meta_parts.append(rng.choice(DELIVERY_BADGES[app]))
        body = [head, price_line, " ".join(meta_parts)]
    return body + [end_line]


def apply_ocr_noise(text: str, rate: float, rng: random.Random) -> str:
    """
    Applies OCR-style noise: each character in `OCR_CONFUSIONS` is swapped with probability
    `rate`, and each space is deleted with probability `rate / 2`. Newlines are untouched.
    A rate of 0.0 returns the text unchanged.
    """
    if rate <= 0.0:
        return text
    out: List[str] = []
    for ch in text:
        if ch in OCR_CONFUSIONS:
            out.append(OCR_CONFUSIONS[ch] if rng.random() < rate else ch)
        elif ch == " ":
            if rng.random() >= rate / 2:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def render_screen(listings: Sequence[List[str]], rng: random.Random) -> str:
    """
    Joins listings into one screen, one line per newline. With probability 0.5 a blank line
    follows each listing except the last.
    """
    lines: List[str] = []
    for i, listing in enumerate(listings):
        lines.extend(listing)
        if i < len(listings) - 1 and rng.random() < 0.5:
            lines.append("")
    return "\n".join(lines)

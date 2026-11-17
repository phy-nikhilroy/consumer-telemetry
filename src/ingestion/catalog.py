"""Larger product catalog built from hand-authored product families.

Each family names a brand, a product line, one of the v1.0 category labels, a base INR price and
variant axes (storage, colour, size). Every combination of variant values is one SKU. The v1.0
``CATALOG`` in ``synthetic_data`` is untouched; its 17 products appear here as families whose
first variant keeps the v1.0 brand, title and base INR price.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import PRICE_TIERS, RANDOM_SEED, USD_TO_INR_RATE  # noqa: E402

# Price uplift per step along an axis: each larger storage or size step costs more.
STORAGE_STEP_UPLIFT = 0.15
SIZE_STEP_UPLIFT = 0.04

_PHONE_COLOURS = ["Black", "Blue", "Silver", "Green", "Pink"]
_STORAGE3 = ["128GB", "256GB", "512GB"]
_STORAGE2 = ["128GB", "256GB"]

# (brand, line, category, base_inr, axes). Axis lists are ordered from cheapest to dearest.
FAMILIES: List[Dict[str, Any]] = []


def _fam(brand: str, line: str, category: str, base_inr: int, **axes: List[str]) -> None:
    """Append one family definition to ``FAMILIES``."""
    FAMILIES.append({"brand": brand, "line": line, "category": category, "base_inr": base_inr, **axes})


# Smartphones: storage variants and colours.
_fam("Apple", "iPhone 15", "Smartphones", 71999, storage=_STORAGE3, colour=_PHONE_COLOURS)
_fam("Apple", "iPhone 15 Pro", "Smartphones", 134900, storage=_STORAGE3, colour=["Natural Titanium", "Blue Titanium", "Black Titanium", "Amber Yellow"])
_fam("Apple", "iPhone 16 Pro", "Smartphones", 119900, storage=["256GB", "512GB", "1TB"], colour=["Natural Titanium", "Desert Titanium", "Sky Blue"])
_fam("Apple", "iPhone SE", "Smartphones", 49900, storage=["64GB", "128GB", "256GB"], colour=["Midnight", "Starlight", "Graphite"])
_fam("Samsung", "Galaxy S24", "Smartphones", 79999, storage=_STORAGE3, colour=["Onyx Black", "Marble Gray", "Cobalt Violet", "Lavender"])
_fam("Samsung", "Galaxy S24 Ultra", "Smartphones", 129999, storage=["256GB", "512GB", "1TB"], colour=["Titanium Gray", "Titanium Black", "Titanium Violet", "Mint"])
_fam("Samsung", "Galaxy M34 5G", "Smartphones", 16999, storage=_STORAGE2, colour=["Midnight Blue", "Prism Silver", "Coral"])
_fam("Samsung", "Galaxy A55", "Smartphones", 39999, storage=_STORAGE2, colour=["Awesome Navy", "Awesome Lilac", "Awesome Iceblue", "Sand"])
_fam("OnePlus", "12R 5G", "Smartphones", 42999, storage=_STORAGE2, colour=["Cool Blue", "Iron Gray", "Forest Green"])
_fam("OnePlus", "12", "Smartphones", 64999, storage=_STORAGE2 + ["512GB"], colour=["Flowy Emerald", "Silky Black", "Rose Gold"])
_fam("OnePlus", "Nord CE 4", "Smartphones", 24999, storage=_STORAGE2, colour=["Dark Chrome", "Celadon Marble", "Crimson"])
_fam("Google", "Pixel 8a", "Smartphones", 52999, storage=["128GB", "256GB"], colour=["Obsidian", "Porcelain", "Bay", "Teal"])
_fam("Google", "Pixel 8", "Smartphones", 75999, storage=_STORAGE2, colour=["Hazel", "Rose", "Obsidian", "Ivory"])
_fam("Google", "Pixel 8 Pro", "Smartphones", 106999, storage=_STORAGE3, colour=["Bay", "Porcelain", "Obsidian", "Copper"])
_fam("Xiaomi", "Redmi Note 13", "Smartphones", 17999, storage=_STORAGE2, colour=["Arctic White", "Prism Gold", "Ice Blue", "Slate"])
_fam("Xiaomi", "Redmi Note 13 Pro", "Smartphones", 25999, storage=_STORAGE2 + ["512GB"], colour=["Midnight Black", "Coral Purple", "Olive"])
_fam("Nothing", "Phone 2a", "Smartphones", 23999, storage=_STORAGE2, colour=["Black", "White", "Milk", "Plum"])
_fam("Motorola", "Edge 50", "Smartphones", 27999, storage=_STORAGE2, colour=["Jungle Green", "Koala Grey", "Pantone Peach", "Peach"])
# Audio
_fam("boAt", "Rockerz 450", "Audio", 1099, colour=["Luscious Black", "Aqua Blue", "Hot Pink", "Navy"])
_fam("boAt", "Airdopes 141", "Audio", 1299, colour=["Bold Black", "Pearl White", "Cool Mint"])
_fam("Sony", "WH-1000XM5", "Audio", 29990, colour=["Black", "Silver", "Midnight Blue"])
_fam("Sony", "WH-1000XM4", "Audio", 22990, colour=["Black", "Silver", "Blue"])
_fam("JBL", "Flip 6", "Audio", 9999, colour=["Black", "Blue", "Red", "Squad", "Teal", "Gray"])
_fam("JBL", "Flip 6 Pro", "Audio", 12999, colour=["Black", "Blue", "Red"])
_fam("Bose", "QuietComfort", "Audio", 26900, colour=["Black", "White Smoke", "Moonstone Blue"])
_fam("Apple", "AirPods Pro", "Audio", 24900, colour=["White"], size=["Standard", "MagSafe Case"])
# Wearables
_fam("Noise", "ColorFit Pulse 3", "Wearables", 1799, colour=["Jet Black", "Rose Pink", "Silver Grey", "Blue"])
_fam("Apple", "Watch Series 9", "Wearables", 41900, size=["41mm", "45mm"], colour=["Midnight", "Starlight", "Pink"])
_fam("Apple", "Watch Ultra", "Wearables", 89900, colour=["Titanium", "Orange Alpine", "Green Alpine", "Blue Ocean"], size=["49mm"])
_fam("Samsung", "Galaxy Watch 6", "Wearables", 29999, size=["40mm", "44mm"], colour=["Graphite", "Gold", "Silver"])
_fam("Garmin", "Forerunner 265", "Wearables", 39990, size=["42mm", "46mm"], colour=["Black", "Aqua", "Whitestone", "Lime"])
_fam("Fitbit", "Charge 6", "Wearables", 14999, colour=["Obsidian", "Porcelain", "Coral"])
# Footwear
_fam("Nike", "Air Monarch IV", "Footwear", 5495, size=["UK 7", "UK 8", "UK 9", "UK 10"], colour=["Black", "White"])
_fam("Nike", "Pegasus 41", "Footwear", 11495, size=["UK 7", "UK 8", "UK 9", "UK 10"], colour=["Black", "Volt"])
_fam("Adidas", "Ultraboost Light", "Footwear", 17999, size=["UK 7", "UK 8", "UK 9", "UK 10"], colour=["Core Black", "Cloud White"])
_fam("Puma", "Smash v2", "Footwear", 2999, size=["UK 7", "UK 8", "UK 9", "UK 10"], colour=["White", "Navy", "Black"])
_fam("Puma", "Smash v2 Mid", "Footwear", 3499, size=["UK 7", "UK 8", "UK 9"], colour=["White", "Black"])
# Storage
_fam("SanDisk", "Extreme Portable SSD", "Storage", 8995, storage=["500GB", "1TB", "2TB"], colour=["Black", "Blue"])
_fam("Samsung", "T7 Portable SSD", "Storage", 7999, storage=["500GB", "1TB", "2TB"], colour=["Gray", "Red"])
_fam("Crucial", "X9 Portable SSD", "Storage", 6999, storage=["1TB", "2TB", "4TB"], colour=["Space Gray"])
_fam("Western Digital", "My Passport", "Storage", 5499, storage=["1TB", "2TB", "4TB"], colour=["Black", "Blue", "Red"])
# Accessories
_fam("Logitech", "MX Master 3S", "Accessories", 8995, colour=["Graphite", "Pale Gray", "Black"])
_fam("Logitech", "MX Master 3S for Mac", "Accessories", 9995, colour=["Space Gray", "Pale Gray"])
_fam("Logitech", "MX Keys", "Accessories", 10995, colour=["Graphite", "Pale Gray"], size=["Full", "Mini"])
_fam("Anker", "PowerCore 20000", "Accessories", 3999, colour=["Black", "White", "Blue"])
_fam("Apple", "MagSafe Charger", "Accessories", 4500, size=["1m", "2m"], colour=["White"])


def _family_id(brand: str, line: str) -> str:
    """Return a stable slug id for a family from its brand and line."""
    return re.sub(r"[^a-z0-9]+", "-", f"{brand} {line}".lower()).strip("-")


def _tier(base_inr: float) -> str:
    """Return the price tier name whose ``PRICE_TIERS`` bounds contain ``base_inr``."""
    for name, (low, high) in PRICE_TIERS.items():
        if low <= base_inr < high:
            return name
    raise ValueError(f"no price tier for {base_inr}")


def build_catalog(seed: int = RANDOM_SEED) -> List[dict]:
    """Expand ``FAMILIES`` into SKU dicts, one per variant combination.

    The expansion is fully deterministic; ``seed`` is accepted for interface parity with the other
    builders and does not change the output. Larger storage and size steps cost more.
    """
    del seed
    skus: List[dict] = []
    for fam in FAMILIES:
        family_id = _family_id(fam["brand"], fam["line"])
        axes = [a for a in ("storage", "colour", "size") if a in fam]
        value_lists = [fam[a] for a in axes]
        for index, combo in enumerate(itertools.product(*value_lists)):
            variant: Dict[str, str] = dict(zip(axes, combo))
            multiplier = 1.0
            if "storage" in variant:
                multiplier += STORAGE_STEP_UPLIFT * fam["storage"].index(variant["storage"])
            if "size" in variant:
                multiplier += SIZE_STEP_UPLIFT * fam["size"].index(variant["size"])
            base_inr = int(round(fam["base_inr"] * multiplier))
            skus.append(
                {
                    "brand": fam["brand"],
                    "title": fam["line"],
                    "specs": list(variant.values()),
                    "base_inr": base_inr,
                    "base_usd": round(base_inr / USD_TO_INR_RATE, 2),
                    "category": fam["category"],
                    "sku_id": f"{family_id}-{index:03d}",
                    "family_id": family_id,
                    "variant": variant,
                    "tier": _tier(base_inr),
                }
            )
    return skus


def main(argv: Optional[List[str]] = None) -> None:
    """Print catalog summary counts, or the full SKU list with ``--dump``."""
    parser = argparse.ArgumentParser(description="Build the product-family catalog.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--dump", action="store_true", help="print every SKU as JSON")
    args = parser.parse_args(argv)
    catalog = build_catalog(args.seed)
    if args.dump:
        print(json.dumps(catalog, indent=2))
    else:
        print(f"{len(FAMILIES)} families, {len(catalog)} SKUs")


if __name__ == "__main__":
    main()

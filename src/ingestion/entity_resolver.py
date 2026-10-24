import hashlib
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

# Stopwords and noise tokens to exclude from canonical keys
STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "for", "with", "by", "from",
    "up", "to", "of", "off", "deal", "day", "limited", "time", "sponsored",
    "best", "seller", "choice", "popular", "pick", "free", "delivery", "buy",
    "now", "add", "cart", "new", "latest", "men", "mens", "women", "womens",
    "unisex", "genuine", "original", "item", "product", "features", "options"
}


def compute_canonical_key(brand: Optional[str], product_title: Optional[str], specs: Optional[List[str]] = None) -> str:
    """
    Constructs a deterministic canonical key:
    lowercase -> remove punctuation & stopwords -> sort remaining unique tokens -> md5 hash.
    """
    parts = []
    if brand:
        parts.append(brand.lower())
    if product_title:
        parts.append(product_title.lower())
    if specs:
        for s in specs:
            parts.append(s.lower())

    combined = " ".join(parts)
    # Strip symbols and punctuation, retain alphanumeric tokens
    tokens = re.findall(r"\b[a-z0-9]+\b", combined)
    filtered = sorted(list(set(t for t in tokens if t not in STOPWORDS and len(t) > 0)))

    if not filtered:
        return "unknown_entity"

    normalized_str = "_".join(filtered)
    return hashlib.md5(normalized_str.encode("utf-8")).hexdigest()[:12]


def compute_canonical_name(brand: Optional[str], product_title: Optional[str]) -> str:
    """
    Returns clean formatted canonical display name.
    """
    b = (brand or "").strip()
    t = (product_title or "").strip()
    if b and t:
        if t.lower().startswith(b.lower()):
            return t
        return f"{b} {t}"
    return b or t or "Unknown Product"


def are_prices_compatible(price1: Optional[float], price2: Optional[float], tolerance_pct: float = 0.20) -> bool:
    """
    Returns True if both prices are within tolerance percentage of each other,
    or if one or both prices are absent.
    """
    if price1 is None or price2 is None:
        return True
    if price1 <= 0 or price2 <= 0:
        return True
    diff = abs(price1 - price2)
    avg = (price1 + price2) / 2.0
    return (diff / avg) <= tolerance_pct


def resolve_entities(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Resolves product records across multiple applications/channels.
    Adds 'canonical_product_id' and 'canonical_name' to each record.
    """
    resolved_products = []

    for item in products:
        prod = dict(item)
        brand = prod.get("brand")
        title = prod.get("product_title")
        specs = prod.get("specs", [])

        # Step 1: Compute primary deterministic canonical key
        canonical_id = compute_canonical_key(brand, title, specs)
        canonical_name = compute_canonical_name(brand, title)

        prod["canonical_product_id"] = canonical_id
        prod["canonical_name"] = canonical_name
        resolved_products.append(prod)

    return resolved_products

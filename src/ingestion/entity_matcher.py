"""
Wording-tolerant product matching.

The v1.0 canonical key hashes the sorted tokens of brand, title, and specs, so two listings of
one product written differently ("256GB" against "256 GB", a dropped colour, a reordered spec)
get different ids, and the warehouse splits one product into several. This module scores a
pair of extracted records with a sentence embedding of the cleaned names and the overlap of
their model-number tokens, vetoes pairs whose brand or storage capacity differ, and keeps a
registry that resolves each new record to an existing product when the score clears
`ENTITY_MATCH_THRESHOLD`.

It also builds labelled pairs from the family catalog (`src/ingestion/catalog.py`): positives
are two renderings of one SKU, hard negatives are other variants of the same family or a
sibling line. Families are split by index parity into tune (even) and test (odd), so the
threshold is chosen on families the reported metrics never see.

Run: .venv/bin/python -m src.ingestion.entity_matcher [--n-pairs 4000] [--seed 42]
writes reports/v1_2_entity_resolution.json and prints the tuned threshold.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from src.config import ENTITY_MATCH_THRESHOLD, RANDOM_SEED, USD_TO_INR_RATE  # noqa: E402
from src.ingestion.entity_resolver import STOPWORDS, are_prices_compatible  # noqa: E402

ENCODER_NAME = "all-MiniLM-L6-v2"

# Interface words that leak into extracted titles from action snippets, delivery badges, and
# app names. Only alphabetic words are taken from the app chrome, so a date or hour in a
# delivery badge never removes a model number.
_EXTRA_UI_WORDS = {
    "view", "details", "detail", "see", "more", "explore", "related", "items", "similar",
    "products", "compare", "prime", "fastest", "tomorrow", "today", "pickup", "shipping",
    "results", "showing", "trending", "sponsored", "result", "stock", "only", "left",
    "list", "wish", "save", "later", "click", "go", "get", "it", "pm", "am",
    "amazon", "flipkart", "walmart", "mrp", "price", "ratings", "rating", "reviews", "review",
    "stars", "star", "out", "free", "deal", "offer", "savings",
}

# Listing text that describes the offer rather than the product: amounts, percentages,
# ratings and their counts, delivery times and dates.
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
          r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b")
_DAY = r"(?:[12]?\d|3[01])"
OFFER_PATTERNS: Tuple[re.Pattern, ...] = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"(?:₹|\$|\brs\.?|\binr\b|\busd\b)\s?\d[\d,]*(?:\.\d+)?",
    r"\d+(?:\.\d+)?\s?%",
    r"\b[1-5]\.\d\s*(?:★|out\s+of\s+5(?:\s+stars?)?)",
    r"[★☆]+",
    r"\(\s*[\d,]+\s*(?:ratings?|reviews?)?\s*\)",
    r"\b\d[\d,]*\s+(?:ratings?|reviews?)\b",
    r"\b\d{1,2}\s?(?:am|pm)\b",
    rf"\b{_DAY}\s+{_MONTH}",
    rf"\b{_MONTH}\s+{_DAY}\b",
    r"\b\d+[- ]days?\b",
    r"\bm\.r\.p\.?:?",
))


def _chrome_words() -> FrozenSet[str]:
    """Alphabetic words of every app's action snippets and delivery badges."""
    from src.ingestion.synthetic_data import ACTION_SNIPPETS, DELIVERY_BADGES

    phrases: List[str] = [p for app in ACTION_SNIPPETS.values() for ps in app.values() for p in ps]
    phrases += [p for ps in DELIVERY_BADGES.values() for p in ps]
    return frozenset(w for w in re.findall(r"[a-z]+", " ".join(phrases).lower()) if len(w) > 1)


UI_WORDS: FrozenSet[str] = frozenset(_EXTRA_UI_WORDS) | _chrome_words()

# Words that name a different product line rather than a variant of one ("Galaxy S24" and
# "Galaxy S24 Ultra"). A name that carries one the other lacks is a different product.
LINE_MODIFIERS: FrozenSet[str] = frozenset({
    "pro", "ultra", "max", "plus", "lite", "mini", "se", "fe",
})

CAPACITY_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(gb|tb)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")

# Weights of the two similarity terms, and the price penalty.
EMBEDDING_WEIGHT = 0.6
MODEL_TOKEN_WEIGHT = 0.4
PRICE_MISMATCH_FACTOR = 0.5
# Applied when the two names disagree on a variant (see `variant_conflict`). This term is
# not in the v1.2 plan's formula: without it, colour and size variants of one family score
# like duplicates (hard-negative false-match rate 0.70 on test families, measured).
VARIANT_CONFLICT_FACTOR = 0.5
PRICE_TOLERANCE_PCT = 0.40

THRESHOLD_GRID: Tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


def _normalise_capacity(text: str) -> str:
    """Rewrites every capacity mention as `<number><unit>` in lower case."""
    return CAPACITY_RE.sub(lambda m: f"{m.group(1)}{m.group(2).lower()}", text)


def _tokens(text: str) -> List[str]:
    """Lower-case word tokens of `text` after capacity normalisation."""
    return TOKEN_RE.findall(_normalise_capacity(text or "").lower())


def model_tokens(text: str) -> FrozenSet[str]:
    """Lower-case tokens that contain a digit (model numbers, capacities, sizes)."""
    return frozenset(t for t in _tokens(text) if any(c.isdigit() for c in t))


def capacity_tokens(text: str) -> FrozenSet[str]:
    """Normalised storage capacities: `256 GB`, `256GB`, and `256gb` all become `256gb`."""
    return frozenset(f"{m.group(1)}{m.group(2).lower()}" for m in CAPACITY_RE.finditer(text or ""))


def strip_offer_text(text: str) -> str:
    """Removes prices, percentages, ratings, rating counts, and delivery times from text."""
    for pattern in OFFER_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def _is_noise(token: str) -> bool:
    """
    Interface words, stopwords, and hyphenated tokens made of them and bare numbers
    ("one-day", "1-click"). A model number such as "wh-1000xm5" is kept.
    """
    if token in UI_WORDS or token in STOPWORDS:
        return True
    parts = [p for p in re.split(r"[-.]", token) if p]
    if len(parts) < 2:
        return False
    wordy = [p for p in parts if p in UI_WORDS or p in STOPWORDS]
    return bool(wordy) and all(p in UI_WORDS or p in STOPWORDS or p.isdigit() for p in parts)


def is_interface_word(word: str) -> bool:
    """
    Whether a whitespace-separated word of listing text is interface chrome, a stopword, or
    bare punctuation (compared lower case, without surrounding punctuation); model numbers
    are never.
    """
    token = word.strip(".,:;!?()[]{}\"'").lower()
    return not token or _is_noise(token)


def record_name(record: dict) -> str:
    """
    The text a record is matched on, lower case, with capacities normalised, interface words
    and stopwords removed, and repeats dropped.

    When the record carries its listing text (`raw_text`), that text is used after the offer
    details are stripped (`strip_offer_text`): the extractor can lose a colour or a shoe size
    from the title, and two variants then extract identically although the listing named
    them. Otherwise the brand, extracted title, and specs are used. A bare number that also
    appears with its unit ("128" next to "128gb") is dropped.
    """
    raw = record.get("raw_text")
    if raw:
        source = strip_offer_text(str(raw))
    else:
        parts = [record.get("brand") or "", record.get("product_title") or ""]
        specs = record.get("specs") or []
        if isinstance(specs, str):
            specs = [s for s in specs.split(",")]
        parts.extend(str(s) for s in specs)
        source = " ".join(parts)
    seen: Dict[str, None] = {}
    for token in _tokens(source):
        if _is_noise(token):
            continue
        seen.setdefault(token, None)
    with_unit = {re.match(r"\d+", t).group(0) for t in seen if re.fullmatch(r"\d+(?:\.\d+)?[a-z]+", t)}
    return " ".join(t for t in seen if not (t.isdigit() and t in with_unit))


def _price_inr(record: dict) -> Optional[float]:
    """The record's price in INR, or None when absent."""
    price = record.get("price")
    if price is None:
        return None
    if (record.get("currency") or "").upper() == "USD":
        return float(price) * USD_TO_INR_RATE
    return float(price)


def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Jaccard overlap; two empty sets count as identical."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def variant_conflict(name_a: str, name_b: str) -> bool:
    """
    Whether two cleaned names disagree on a variant or a product line.

    Capacity tokens are left to the capacity veto. Of the remaining tokens, a conflict is
    both names carrying a token the other lacks (Black against Silver, UK 7 against UK 8),
    or either carrying a line modifier the other lacks (Pro, Ultra). One name merely having
    extra tokens is not a conflict: a listing often omits a colour or a spec.
    """
    tokens_a = set(name_a.split()) - capacity_tokens(name_a)
    tokens_b = set(name_b.split()) - capacity_tokens(name_b)
    only_a, only_b = tokens_a - tokens_b, tokens_b - tokens_a
    if only_a and only_b:
        return True
    return bool((only_a | only_b) & LINE_MODIFIERS)


class _EmbeddingCache:
    """Caches one normalised embedding per name for a given encoder."""

    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self.vectors: Dict[str, np.ndarray] = {}

    def get(self, name: str) -> np.ndarray:
        vec = self.vectors.get(name)
        if vec is None:
            vec = np.asarray(self.encoder.encode([name], normalize_embeddings=True)[0], dtype=np.float32)
            self.vectors[name] = vec
        return vec

    def warm(self, names: Iterable[str]) -> None:
        missing = sorted({n for n in names if n not in self.vectors})
        if missing:
            vecs = self.encoder.encode(missing, normalize_embeddings=True, batch_size=256)
            for n, v in zip(missing, vecs):
                self.vectors[n] = np.asarray(v, dtype=np.float32)


_CACHES: Dict[int, _EmbeddingCache] = {}


def _cache_for(encoder: Any) -> _EmbeddingCache:
    cache = _CACHES.get(id(encoder))
    if cache is None or cache.encoder is not encoder:
        cache = _EmbeddingCache(encoder)
        _CACHES[id(encoder)] = cache
    return cache


@lru_cache(maxsize=1)
def default_encoder() -> Any:
    """The sentence encoder used when none is supplied (loaded once)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(ENCODER_NAME)


def pair_score(a: dict, b: dict, encoder: Any) -> float:
    """
    Scores how likely two extracted records are the same product, in [0, 1].

    0.0 when the brands differ (case-insensitive) or when both name capacities and the sets
    differ. Otherwise 0.6 * cosine of the name embeddings + 0.4 * Jaccard of model tokens,
    halved when both prices are present and differ by more than 40 percent, and halved
    again when the names disagree on a variant (`variant_conflict`).
    """
    brand_a = (a.get("brand") or "").strip().lower()
    brand_b = (b.get("brand") or "").strip().lower()
    if brand_a != brand_b:
        return 0.0
    name_a, name_b = record_name(a), record_name(b)
    cap_a, cap_b = capacity_tokens(name_a), capacity_tokens(name_b)
    if cap_a and cap_b and cap_a != cap_b:
        return 0.0
    cache = _cache_for(encoder)
    cosine = float(np.dot(cache.get(name_a), cache.get(name_b)))
    score = EMBEDDING_WEIGHT * cosine + MODEL_TOKEN_WEIGHT * _jaccard(model_tokens(name_a), model_tokens(name_b))
    price_a, price_b = _price_inr(a), _price_inr(b)
    if price_a is not None and price_b is not None and not are_prices_compatible(
        price_a, price_b, tolerance_pct=PRICE_TOLERANCE_PCT
    ):
        score *= PRICE_MISMATCH_FACTOR
    if variant_conflict(name_a, name_b):
        score *= VARIANT_CONFLICT_FACTOR
    return float(min(1.0, max(0.0, score)))


def _v1_key(record: dict) -> str:
    """The v1.0 exact canonical key of a record."""
    from src.ingestion.entity_resolver import compute_canonical_key

    key = record.get("canonical_product_id")
    if key:
        return str(key)
    specs = record.get("specs") or []
    if isinstance(specs, str):
        specs = [s.strip() for s in specs.split(",") if s.strip()]
    return compute_canonical_key(record.get("brand"), record.get("product_title"), specs)


_ENTRY_FIELDS = ("brand", "product_title", "specs", "price", "currency", "raw_text")


class ProductRegistry:
    """
    Resolves records to canonical product ids, tolerating different wording.

    An exact v1.0 key already seen returns its id at once. Otherwise the record is scored
    against every registered product of the same brand; the best id is returned when its
    score reaches the threshold, and the record's v1.0 key is remembered as an alias of it.
    A record that matches nothing is registered as a new product under its own v1.0 key.
    """

    def __init__(self, threshold: float = ENTITY_MATCH_THRESHOLD, encoder: Any = None) -> None:
        self.threshold = float(threshold)
        self._encoder = encoder
        self._entries: Dict[str, dict] = {}
        self._aliases: Dict[str, str] = {}
        self._by_brand: Dict[str, List[str]] = {}

    @property
    def encoder(self) -> Any:
        if self._encoder is None:
            self._encoder = default_encoder()
        return self._encoder

    def _register(self, product_id: str, record: dict) -> None:
        entry = {k: record.get(k) for k in _ENTRY_FIELDS}
        if isinstance(entry.get("specs"), tuple):
            entry["specs"] = list(entry["specs"])
        self._entries[product_id] = entry
        self._aliases[product_id] = product_id
        self._by_brand.setdefault((entry.get("brand") or "").strip().lower(), []).append(product_id)

    def resolve(self, record: dict) -> str:
        """Returns the canonical product id for `record`, registering it when new."""
        key = _v1_key(record)
        hit = self._aliases.get(key)
        if hit is not None:
            return hit
        brand = (record.get("brand") or "").strip().lower()
        best_id, best_score = None, -1.0
        for product_id in self._by_brand.get(brand, []):
            score = pair_score(record, self._entries[product_id], self.encoder)
            if score > best_score:
                best_id, best_score = product_id, score
        if best_id is not None and best_score >= self.threshold:
            self._aliases[key] = best_id
            return best_id
        self._register(key, record)
        return key

    def __len__(self) -> int:
        return len(self._entries)

    def to_records(self) -> List[dict]:
        """Serialisable state: one dict per product with its fields and aliases."""
        aliases: Dict[str, List[str]] = {pid: [] for pid in self._entries}
        for alias, pid in self._aliases.items():
            if alias != pid:
                aliases[pid].append(alias)
        return [
            {"canonical_product_id": pid, **entry, "aliases": sorted(aliases[pid])}
            for pid, entry in self._entries.items()
        ]

    @classmethod
    def from_records(
        cls, records: Sequence[dict], threshold: float = ENTITY_MATCH_THRESHOLD, encoder: Any = None
    ) -> "ProductRegistry":
        """Rebuilds a registry from `to_records()` output."""
        registry = cls(threshold=threshold, encoder=encoder)
        for rec in records:
            pid = rec["canonical_product_id"]
            registry._register(pid, rec)
            for alias in rec.get("aliases", []):
                registry._aliases[alias] = pid
        return registry


# ---------------------------------------------------------------------------------------
# Labelled pairs and evaluation
# ---------------------------------------------------------------------------------------

APPS = ("amazon_in", "amazon_us", "flipkart", "walmart")
HARD_NEGATIVE_SHARE = 0.6


def _is_sibling(line_a: str, line_b: str) -> bool:
    """Lines of one brand that differ by one added or substituted token."""
    ta, tb = line_a.lower().split(), line_b.lower().split()
    if ta == tb:
        return False
    sa, sb = set(ta), set(tb)
    if abs(len(ta) - len(tb)) == 1:
        return len(sa ^ sb) == 1
    if len(ta) == len(tb):
        return len(sa - sb) == 1 and len(sb - sa) == 1
    return False


def family_split(catalog: Sequence[dict]) -> Dict[str, str]:
    """
    Maps each family id to `tune` or `test`.

    Families are first joined into groups of sibling lines of one brand (`_is_sibling`), so
    that a sibling hard negative never straddles the split; a family with no sibling is its
    own group. Groups in order of first appearance alternate: even index tune, odd test.
    Siblings are usually adjacent in the catalog, so a plain per-family parity would put
    every sibling pair across the split and leave no sibling negatives at all.
    """
    order: Dict[str, int] = {}
    line_brand: Dict[str, Tuple[str, str]] = {}
    for sku in catalog:
        if sku["family_id"] not in order:
            order[sku["family_id"]] = len(order)
            line_brand[sku["family_id"]] = (sku["brand"], sku["title"])
    parent = {fid: fid for fid in order}

    def root(fid: str) -> str:
        while parent[fid] != fid:
            parent[fid] = parent[parent[fid]]
            fid = parent[fid]
        return fid

    fids = list(order)
    for i, fa in enumerate(fids):
        for fb in fids[i + 1:]:
            (ba, la), (bb, lb) = line_brand[fa], line_brand[fb]
            if ba == bb and _is_sibling(la, lb):
                ra, rb = root(fa), root(fb)
                if ra != rb:
                    parent[max(ra, rb, key=order.get)] = min(ra, rb, key=order.get)
    group_index: Dict[str, int] = {}
    for fid in fids:
        group_index.setdefault(root(fid), len(group_index))
    return {fid: ("tune" if group_index[root(fid)] % 2 == 0 else "test") for fid in fids}


def _shows(text: str, value: str) -> bool:
    """Whether a rendered listing shows a variant value (ignoring case and spaces)."""
    return re.sub(r"\s+", "", value.lower()) in re.sub(r"\s+", "", text.lower())


MAX_RENDER_ATTEMPTS = 25


def _render_record(sku: dict, rng: random.Random, cache: Dict[str, dict],
                   must_show: Sequence[str] = ()) -> dict:
    """
    Renders one listing of `sku` and extracts it with `process` (no registry).

    With `must_show`, the listing is re-rendered (up to `MAX_RENDER_ATTEMPTS` times) until it
    shows every given variant value, so a hard negative keeps the attribute that makes it a
    different SKU instead of losing it to a random spec drop.
    """
    from src.ingestion.pipeline import process
    from src.ingestion.screen_generator import render_listing

    for _ in range(MAX_RENDER_ATTEMPTS):
        lines = render_listing(
            sku,
            app=rng.choice(APPS),
            action=rng.choice(["view", "cart", "buy", "search", "wishlist"]),
            discount_pct=rng.choice([0, 0, 10, 20]),
            rating=round(rng.uniform(3.5, 4.9), 1),
            rng=rng,
        )
        text = " ".join(lines)
        if all(_shows(text, v) for v in must_show):
            break
    record = cache.get(text)
    if record is None:
        record = process(text).model_dump(
            include={"raw_text", "brand", "product_title", "specs", "price", "currency", "canonical_product_id"}
        )
        cache[text] = record
    return dict(record)


def build_labelled_pairs(catalog: Sequence[dict], seed: int = RANDOM_SEED, n_pairs: int = 4000) -> List[dict]:
    """
    Builds labelled record pairs from the family catalog.

    Half the pairs are positives (two renderings of one SKU). Of the negatives,
    `HARD_NEGATIVE_SHARE` are hard: another variant of the same family, or a SKU of a sibling
    line; the rest are random SKUs of the same brand or category in another family. Every
    pair lies within one split, so tune and test share no family. Each pair is a dict with
    `a`, `b` (extracted records), `label` (1 same SKU), `kind` (`positive`, `hard`,
    `random`), `split`, and `families`.

    Each side of a same-family hard negative is rendered so that it shows the variant
    values on which the two SKUs differ. Positives keep the random spec drops. Without this,
    a dropped colour or size can make two different SKUs extract to identical text, which no
    matcher (and not the v1.0 key either) can tell apart.
    """
    rng = random.Random(seed)
    split_of = family_split(catalog)
    by_family: Dict[str, List[dict]] = {}
    for sku in catalog:
        by_family.setdefault(sku["family_id"], []).append(sku)
    line_of = {fid: skus[0]["title"] for fid, skus in by_family.items()}
    brand_of = {fid: skus[0]["brand"] for fid, skus in by_family.items()}
    category_of = {fid: skus[0]["category"] for fid, skus in by_family.items()}

    siblings: Dict[str, List[str]] = {fid: [] for fid in by_family}
    for fa in by_family:
        for fb in by_family:
            if fa != fb and brand_of[fa] == brand_of[fb] and split_of[fa] == split_of[fb] \
                    and _is_sibling(line_of[fa], line_of[fb]):
                siblings[fa].append(fb)

    cache: Dict[str, dict] = {}
    pairs: List[dict] = []
    families = sorted(by_family)
    n_pos = n_pairs // 2
    n_neg = n_pairs - n_pos
    n_hard = int(round(n_neg * HARD_NEGATIVE_SHARE))

    def add(sku_a: dict, sku_b: dict, label: int, kind: str) -> None:
        differ = [axis for axis in sku_a.get("variant", {})
                  if kind == "hard" and sku_a["variant"].get(axis) != sku_b.get("variant", {}).get(axis)]
        pairs.append({
            "a": _render_record(sku_a, rng, cache, [sku_a["variant"][x] for x in differ]),
            "b": _render_record(sku_b, rng, cache, [sku_b["variant"][x] for x in differ if x in sku_b["variant"]]),
            "label": label,
            "kind": kind,
            "split": split_of[sku_a["family_id"]],
            "families": sorted({sku_a["family_id"], sku_b["family_id"]}),
            "skus": [sku_a["sku_id"], sku_b["sku_id"]],
        })

    for _ in range(n_pos):
        sku = rng.choice(by_family[rng.choice(families)])
        add(sku, sku, 1, "positive")

    hard_families = [f for f in families if len(by_family[f]) > 1 or siblings[f]]
    for _ in range(n_hard):
        fid = rng.choice(hard_families)
        use_sibling = bool(siblings[fid]) and (len(by_family[fid]) == 1 or rng.random() < 0.5)
        sku_a = rng.choice(by_family[fid])
        if use_sibling:
            sku_b = rng.choice(by_family[rng.choice(siblings[fid])])
        else:
            sku_b = rng.choice([s for s in by_family[fid] if s["sku_id"] != sku_a["sku_id"]])
        add(sku_a, sku_b, 0, "hard")

    for _ in range(n_neg - n_hard):
        fid = rng.choice(families)
        pool = [
            f for f in families
            if f != fid and split_of[f] == split_of[fid]
            and (brand_of[f] == brand_of[fid] or category_of[f] == category_of[fid])
        ]
        if not pool:
            continue
        add(rng.choice(by_family[fid]), rng.choice(by_family[rng.choice(pool)]), 0, "random")
    return pairs


def _metrics(labels: Sequence[int], predicted: Sequence[bool], kinds: Sequence[str]) -> Dict[str, float]:
    """Precision, recall, F1, and the false-match rate on hard negatives."""
    tp = sum(1 for y, p in zip(labels, predicted) if y == 1 and p)
    fp = sum(1 for y, p in zip(labels, predicted) if y == 0 and p)
    fn = sum(1 for y, p in zip(labels, predicted) if y == 1 and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hard = [p for p, k in zip(predicted, kinds) if k == "hard"]
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hard_negative_false_match_rate": round(sum(hard) / len(hard), 4) if hard else 0.0,
        "n_pairs": len(labels),
        "n_hard_negatives": len(hard),
    }


def score_pairs(pairs: Sequence[dict], encoder: Any) -> List[float]:
    """`pair_score` of every pair, with the name embeddings computed in one batch."""
    _cache_for(encoder).warm(record_name(r) for p in pairs for r in (p["a"], p["b"]))
    return [pair_score(p["a"], p["b"], encoder) for p in pairs]


def tune_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    """The grid value maximising F1; ties go to the higher threshold."""
    best, best_f1 = THRESHOLD_GRID[-1], -1.0
    for thr in THRESHOLD_GRID:
        f1 = _metrics(labels, [s >= thr for s in scores], ["random"] * len(labels))["f1"]
        if f1 >= best_f1:
            best, best_f1 = thr, f1
    return best


def evaluate(pairs: Sequence[dict], encoder: Any) -> Dict[str, Any]:
    """Tunes the threshold on tune pairs and reports matcher and v1.0-key metrics on test pairs."""
    scores = score_pairs(pairs, encoder)
    tune = [i for i, p in enumerate(pairs) if p["split"] == "tune"]
    test = [i for i, p in enumerate(pairs) if p["split"] == "test"]
    threshold = tune_threshold([scores[i] for i in tune], [pairs[i]["label"] for i in tune])

    def block(idx: List[int], thr: float) -> Dict[str, Any]:
        labels = [pairs[i]["label"] for i in idx]
        kinds = [pairs[i]["kind"] for i in idx]
        exact = [pairs[i]["a"]["canonical_product_id"] == pairs[i]["b"]["canonical_product_id"] for i in idx]
        return {
            "matcher": _metrics(labels, [scores[i] >= thr for i in idx], kinds),
            "v1_exact_key": _metrics(labels, exact, kinds),
        }

    return {
        "threshold": threshold,
        "threshold_grid": list(THRESHOLD_GRID),
        "tune": block(tune, threshold),
        "test": block(test, threshold),
        "n_pairs": len(pairs),
        "n_families": {"tune": len({f for i in tune for f in pairs[i]["families"]}),
                       "test": len({f for i in test for f in pairs[i]["families"]})},
    }


def main(argv: Optional[List[str]] = None) -> None:
    """Builds the pairs, tunes the threshold, and writes the entity-resolution report."""
    parser = argparse.ArgumentParser(description="Evaluate the wording-tolerant product matcher.")
    parser.add_argument("--n-pairs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-report", action="store_true", help="print results without writing the report")
    args = parser.parse_args(argv)

    from src.ingestion.catalog import build_catalog

    pairs = build_labelled_pairs(build_catalog(args.seed), seed=args.seed, n_pairs=args.n_pairs)
    results = evaluate(pairs, default_encoder())
    print(f"threshold {results['threshold']}")
    for split in ("tune", "test"):
        print(split, "matcher", results[split]["matcher"])
        print(split, "v1 key ", results[split]["v1_exact_key"])
    if not args.no_report:
        from src.release.reporting import write_report

        path = write_report("v1_2_entity_resolution", "1.2", results, seed=args.seed)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

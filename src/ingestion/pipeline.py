"""
Upstream ingestion pipeline: raw screen text to a validated, resolved product record.

Wires the regex extractors, the linguistic brand and title extractors, the taxonomy
classifier, and the entity resolver into `process()`, and scores how far each record can
be trusted with `calculate_extraction_confidence()`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import polars as pl
from pydantic import BaseModel, Field, field_validator

from src.ingestion.regex_extractor import extract_price, extract_rating, extract_action
from src.config import TAXONOMY_CONFIDENT_SIMILARITY, TAXONOMY_MIN_SIMILARITY
from src.ingestion.ner_extractor import (
    BRAND_SOURCE_LEADING_MATCH,
    BRAND_SOURCE_LEADING_TOKEN,
    BRAND_SOURCE_MATCH,
    extract_brand_with_source,
    extract_price_mentions,
    extract_price_mentions_batch,
    extract_product_title,
    extract_specs,
)
from src.ingestion.taxonomy_classifier import METHOD_SUPERVISED, classify_product, warm_taxonomy_queries
from src.ingestion.entity_resolver import compute_canonical_key, compute_canonical_name
from src.ingestion.confidence_model import predict_confidence
from src.ingestion.segmenter import segment_screen

if TYPE_CHECKING:
    from src.ingestion.entity_matcher import ProductRegistry


class ExtractedProduct(BaseModel):
    raw_text: str
    app_source: str = "unknown"
    brand: Optional[str] = None
    product_title: Optional[str] = None
    canonical_name: str
    canonical_product_id: str
    specs: List[str] = Field(default_factory=list)
    category_l1: str = "Unknown"
    category_l2: str = "Unknown"
    full_category: str = "Unknown"
    price: Optional[float] = None
    currency: Optional[str] = None
    mrp: Optional[float] = None
    discount_pct: Optional[int] = 0
    rating: Optional[float] = None
    num_ratings: Optional[int] = None
    action: str = "view"
    brand_source: Optional[str] = None
    category_similarity: float = 0.0
    category_probability: float = 0.0
    category_model_margin: float = 0.0
    category_method: Optional[str] = None
    price_verified: bool = False
    extraction_confidence: float = 0.0
    is_focal: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Coerces an unrecognized action to "view" rather than rejecting the record."""
        valid_actions = {"view", "cart", "buy", "search", "wishlist"}
        return v if v in valid_actions else "view"


# How much each field contributes to the overall confidence score.
CONFIDENCE_WEIGHTS: Dict[str, float] = {
    "brand": 0.25,
    "category": 0.25,
    "price": 0.25,
    "product_title": 0.15,
    "currency": 0.05,
    "rating": 0.05,
}

# How far each kind of brand evidence can be trusted. The values follow the precision
# each source achieved on real out-of-domain listings (week_3_real_data_validation.md):
# a known brand leading the title is right most of the time, the leading-word fallback
# somewhat less, and a known brand found only mid-title is usually the device an
# accessory fits rather than its maker.
BRAND_EVIDENCE: Dict[Optional[str], float] = {
    BRAND_SOURCE_LEADING_MATCH: 0.9,
    BRAND_SOURCE_LEADING_TOKEN: 0.7,
    BRAND_SOURCE_MATCH: 0.2,
    None: 0.0,
}

# A regex price that spaCy's statistical NER also recognizes as money counts fully; a
# regex price with no independent confirmation counts for less.
PRICE_VERIFIED_EVIDENCE = 1.0
PRICE_UNVERIFIED_EVIDENCE = 0.7


def category_evidence(similarity: float, abstained: bool, method: Optional[str] = None) -> float:
    """
    Maps the taxonomy score onto [0, 1]. A supervised-model answer is already a probability
    and is used as it is. A prototype similarity is zero at or below the abstain floor and
    rises linearly to one at `TAXONOMY_CONFIDENT_SIMILARITY`.
    """
    if abstained:
        return 0.0
    if method == METHOD_SUPERVISED:
        return float(min(1.0, max(0.0, similarity)))
    span = TAXONOMY_CONFIDENT_SIMILARITY - TAXONOMY_MIN_SIMILARITY
    if span <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (similarity - TAXONOMY_MIN_SIMILARITY) / span)))


def price_is_verified(price: Optional[float], ner_amounts: List[float]) -> bool:
    """Returns whether an independent NER money mention agrees with the regex price."""
    if price is None:
        return False
    return any(abs(amount - float(price)) < 0.01 for amount in ner_amounts)


def model_probability_and_margin(taxonomy_info: Dict[str, Any]) -> tuple[float, float]:
    """
    The supervised model's probability of the category actually assigned, and its top-minus-
    second margin, on one scale whichever path answered. Zero for the prototype-only path,
    for an empty title, and for a fallback whose label the model does not know.
    """
    probability = float(taxonomy_info.get("model_assigned_probability", 0.0))
    if taxonomy_info.get("method") == METHOD_SUPERVISED:
        return probability, float(taxonomy_info.get("margin", 0.0))
    return probability, float(taxonomy_info.get("model_margin", 0.0))


def calculate_extraction_confidence(extracted: Dict[str, Any], artifact_path: Optional[Path] = None) -> float:
    """
    Returns the learned model probability when the confidence artifact exists (at
    `artifact_path`, default `config.CONFIDENCE_ARTIFACT_PATH`), otherwise the hand-weighted
    score. A record with no extracted evidence at all scores 0.0 either way.
    """
    hand = calculate_hand_weighted_confidence(extracted)
    if hand == 0.0 and not extracted.get("brand") and extracted.get("price") is None:
        return 0.0
    learned = predict_confidence(extracted, artifact_path)
    return hand if learned is None else learned


def product_to_evidence(product: "ExtractedProduct") -> Dict[str, Any]:
    """The confidence inputs of an extracted product, with the category margin."""
    return {
        "brand": product.brand,
        "brand_source": product.brand_source,
        "product_title": product.product_title,
        "price": product.price,
        "price_verified": product.price_verified,
        "currency": product.currency,
        "rating": product.rating,
        "category_l2": product.category_l2,
        "full_category": product.full_category,
        "category_similarity": product.category_similarity,
        "category_probability": product.category_probability,
        "category_model_margin": product.category_model_margin,
        "category_abstained": product.category_l2 == "Unknown",
        "category_method": product.category_method,
    }


def calculate_hand_weighted_confidence(extracted: Dict[str, Any]) -> float:
    """
    Scores how far an extracted record can be trusted, from 0.0 to 1.0.

    Each field contributes its weight multiplied by the strength of the evidence behind
    it, not merely whether it is populated. The previous score counted populated fields,
    so a brand that was confidently wrong scored exactly like a correct one; on real
    listings it averaged 0.90 while brand accuracy was 1 percent.

    Recognized keys: `brand`, `brand_source`, `product_title`, `price`,
    `price_verified`, `currency`, `rating`, `category_l2`, `category_similarity`,
    `category_abstained`, and `category_method`.
    """
    score = 0.0

    if extracted.get("brand"):
        score += CONFIDENCE_WEIGHTS["brand"] * BRAND_EVIDENCE.get(extracted.get("brand_source"), 0.0)

    if extracted.get("product_title"):
        score += CONFIDENCE_WEIGHTS["product_title"]

    if extracted.get("price") is not None:
        evidence = PRICE_VERIFIED_EVIDENCE if extracted.get("price_verified") else PRICE_UNVERIFIED_EVIDENCE
        score += CONFIDENCE_WEIGHTS["price"] * evidence
        if extracted.get("currency"):
            score += CONFIDENCE_WEIGHTS["currency"]

    if extracted.get("rating") is not None:
        score += CONFIDENCE_WEIGHTS["rating"]

    category = extracted.get("category_l2")
    abstained = bool(extracted.get("category_abstained")) or not category or category == "Unknown"
    score += CONFIDENCE_WEIGHTS["category"] * category_evidence(
        float(extracted.get("category_similarity") or 0.0), abstained, extracted.get("category_method")
    )

    return round(score, 4)


def _linguistic_fields(raw_text: str) -> tuple:
    """Brand, brand source, title, and specs of one text."""
    brand, brand_source = extract_brand_with_source(raw_text)
    return brand, brand_source, extract_product_title(raw_text, brand=brand), extract_specs(raw_text)


def process(
    raw_text: str,
    app_source: str = "unknown",
    timestamp: Optional[datetime] = None,
    registry: Optional["ProductRegistry"] = None,
) -> ExtractedProduct:
    """
    Wires together regex, NER, taxonomy classifier, and entity resolver.
    Returns a validated ExtractedProduct Pydantic instance.

    With a `registry`, the canonical product id comes from `registry.resolve`, so differently
    worded listings of one product share an id. Without one, the v1.0 exact key is used.
    """
    return _build_product(raw_text, app_source, timestamp, registry, extract_price_mentions(raw_text))


def _build_product(
    raw_text: str,
    app_source: str,
    timestamp: Optional[datetime],
    registry: Optional["ProductRegistry"],
    ner_amounts: List[float],
    extracted: Optional[tuple] = None,
) -> ExtractedProduct:
    """The body of `process`, with the NER money amounts (and optionally the extractor output) supplied."""
    # 1. Deterministic regex extraction
    price_info = extract_price(raw_text)
    rating_info = extract_rating(raw_text)
    action = extract_action(raw_text)

    # 2. Linguistic NER extraction
    if extracted is None:
        extracted = _linguistic_fields(raw_text)
    brand, brand_source, product_title, specs = extracted
    price_verified = price_is_verified(price_info.get("price"), ner_amounts)

    # 3. Semantic taxonomy classification
    taxonomy_info = classify_product(product_title or "", brand=brand)

    # 4. Entity resolution
    canonical_id = compute_canonical_key(brand, product_title, specs)
    canonical_name = compute_canonical_name(brand, product_title)
    if registry is not None:
        canonical_id = registry.resolve({
            "raw_text": raw_text,
            "brand": brand,
            "product_title": product_title,
            "specs": specs,
            "price": price_info.get("price"),
            "currency": price_info.get("currency"),
            "canonical_product_id": canonical_id,
        })

    # 5. Extraction confidence
    category_similarity = float(taxonomy_info.get("confidence", 0.0))
    category_probability, category_model_margin = model_probability_and_margin(taxonomy_info)
    extraction_data = {
        "brand": brand,
        "brand_source": brand_source,
        "product_title": product_title,
        "price": price_info.get("price"),
        "price_verified": price_verified,
        "currency": price_info.get("currency"),
        "rating": rating_info.get("rating"),
        "category_l2": taxonomy_info.get("category_l2"),
        "category_similarity": category_similarity,
        "category_probability": category_probability,
        "category_model_margin": category_model_margin,
        "category_abstained": taxonomy_info.get("abstained", False),
        "category_method": taxonomy_info.get("method"),
    }
    confidence = calculate_extraction_confidence(extraction_data)

    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    return ExtractedProduct(
        raw_text=raw_text,
        app_source=app_source,
        brand=brand,
        product_title=product_title,
        canonical_name=canonical_name,
        canonical_product_id=canonical_id,
        specs=specs,
        category_l1=taxonomy_info.get("category_l1", "Unknown"),
        category_l2=taxonomy_info.get("category_l2", "Unknown"),
        full_category=taxonomy_info.get("full_category", "Unknown"),
        price=price_info.get("price"),
        currency=price_info.get("currency"),
        mrp=price_info.get("mrp"),
        discount_pct=price_info.get("discount_pct", 0),
        rating=rating_info.get("rating"),
        num_ratings=rating_info.get("num_ratings"),
        action=action,
        brand_source=brand_source,
        category_similarity=category_similarity,
        category_probability=category_probability,
        category_model_margin=category_model_margin,
        category_method=taxonomy_info.get("method"),
        price_verified=price_verified,
        extraction_confidence=confidence,
        timestamp=timestamp
    )


FOCAL_ACTIONS = ("buy", "cart")


def process_screen(
    raw_text: str,
    app_source: str = "unknown",
    timestamp: Optional[datetime] = None,
    registry: Optional["ProductRegistry"] = None,
) -> List[ExtractedProduct]:
    """
    Extracts every listing on a screen, one `ExtractedProduct` per segment.

    The screen is split with `segment_screen`, and each segment goes through `process`. All
    products share one timestamp, because one screen is captured at one moment. Exactly one
    product is marked `is_focal`: the first whose action is buy or cart, otherwise the first.
    A v1.0 one-line string yields a single product equal to `process(raw_text)` apart from
    `is_focal`. A `registry` is passed through to `process`. A multi-line screen with no text yields an empty list.
    """
    segments = segment_screen(raw_text)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    products = [
        process(segment, app_source=app_source, timestamp=timestamp, registry=registry) for segment in segments
    ]
    if not products:
        return products
    focal = next((i for i, p in enumerate(products) if p.action in FOCAL_ACTIONS), 0)
    products[focal].is_focal = True
    return products


def process_batch(
    raw_texts: Sequence[str],
    app_sources: Optional[Sequence[str]] = None,
    timestamps: Optional[Sequence[Optional[datetime]]] = None,
    registry: Optional["ProductRegistry"] = None,
    batch_size: int = 256,
) -> List[ExtractedProduct]:
    """
    Runs many texts through the pipeline, equal to `[process(t, ...) for t in raw_texts]`.

    The spaCy money pass goes through `nlp.pipe` and the taxonomy queries are encoded in
    batches into the shared query cache before each record is classified. `app_sources`
    defaults to "unknown" and `timestamps` to the current time; a supplied sequence must
    match `raw_texts` in length or `ValueError` is raised.
    """
    n = len(raw_texts)
    if app_sources is not None and len(app_sources) != n:
        raise ValueError(f"app_sources has {len(app_sources)} items for {n} texts")
    if timestamps is not None and len(timestamps) != n:
        raise ValueError(f"timestamps has {len(timestamps)} items for {n} texts")
    if n == 0:
        return []
    texts = list(raw_texts)
    amounts = extract_price_mentions_batch(texts, batch_size=batch_size)
    fields = [_linguistic_fields(t) for t in texts]
    warm_taxonomy_queries([(f[2], f[0]) for f in fields], batch_size=batch_size)
    return [
        _build_product(
            texts[i],
            app_sources[i] if app_sources is not None else "unknown",
            timestamps[i] if timestamps is not None else None,
            registry,
            amounts[i],
            fields[i],
        )
        for i in range(n)
    ]


def run_pipeline(input_jsonl_path: str, output_parquet_path: str) -> pl.DataFrame:
    """
    Executes the batch upstream ingestion pipeline:
    reads raw screen dumps -> validates records -> outputs Parquet table.
    """
    input_file = Path(input_jsonl_path)
    output_file = Path(output_parquet_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    extracted_records: List[Dict[str, Any]] = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            raw_text = entry.get("raw_text", "")
            app_source = entry.get("app", "unknown")
            ts_str = entry.get("timestamp")

            dt = None
            if ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str)
                except ValueError:
                    dt = None

            product = process(raw_text, app_source=app_source, timestamp=dt)
            record_dict = product.model_dump()
            # Convert specs list to comma-separated string for high-performance tabular storage
            record_dict["specs"] = ", ".join(record_dict["specs"])
            extracted_records.append(record_dict)

    df = pl.DataFrame(extracted_records)
    df.write_parquet(output_file)
    return df

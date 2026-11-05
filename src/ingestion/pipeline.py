from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import polars as pl
from pydantic import BaseModel, Field, field_validator

from src.ingestion.regex_extractor import extract_price, extract_rating, extract_action
from src.ingestion.ner_extractor import extract_brand, extract_product_title, extract_specs
from src.ingestion.taxonomy_classifier import classify_product
from src.ingestion.entity_resolver import compute_canonical_key, compute_canonical_name


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
    extraction_confidence: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Coerces an unrecognized action to "view" rather than rejecting the record."""
        valid_actions = {"view", "cart", "buy", "search", "wishlist"}
        return v if v in valid_actions else "view"


def calculate_extraction_confidence(extracted: Dict[str, Any]) -> float:
    """
    Computes a normalized confidence score (0.0 to 1.0) based on extraction coverage.
    """
    weights = {
        "brand": 0.20,
        "product_title": 0.25,
        "price": 0.25,
        "currency": 0.10,
        "rating": 0.10,
        "category": 0.10,
    }

    score = 0.0
    if extracted.get("brand"):
        score += weights["brand"]
    if extracted.get("product_title"):
        score += weights["product_title"]
    if extracted.get("price") is not None:
        score += weights["price"]
    if extracted.get("currency"):
        score += weights["currency"]
    if extracted.get("rating") is not None:
        score += weights["rating"]
    if extracted.get("category_l2") and extracted.get("category_l2") != "Unknown":
        score += weights["category"]

    return round(score, 4)


def process(
    raw_text: str,
    app_source: str = "unknown",
    timestamp: Optional[datetime] = None
) -> ExtractedProduct:
    """
    Wires together regex, NER, taxonomy classifier, and entity resolver.
    Returns a validated ExtractedProduct Pydantic instance.
    """
    # 1. Deterministic regex extraction
    price_info = extract_price(raw_text)
    rating_info = extract_rating(raw_text)
    action = extract_action(raw_text)

    # 2. Linguistic NER extraction
    brand = extract_brand(raw_text)
    product_title = extract_product_title(raw_text, brand=brand)
    specs = extract_specs(raw_text)

    # 3. Semantic taxonomy classification
    taxonomy_info = classify_product(product_title or "", brand=brand)

    # 4. Entity resolution
    canonical_id = compute_canonical_key(brand, product_title, specs)
    canonical_name = compute_canonical_name(brand, product_title)

    # 5. Extraction confidence
    extraction_data = {
        "brand": brand,
        "product_title": product_title,
        "price": price_info.get("price"),
        "currency": price_info.get("currency"),
        "rating": rating_info.get("rating"),
        "category_l2": taxonomy_info.get("category_l2")
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
        extraction_confidence=confidence,
        timestamp=timestamp
    )


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

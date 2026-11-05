"""
Out-of-domain benchmark of the upstream extraction chain against real product data.

The synthetic generator and the extractors were written together, so measuring the
extractors against synthetic screen dumps mostly measures self-consistency. This module
runs the same chain against real Amazon product metadata, using the dataset's own
structured fields as ground truth.

One honest caveat about the setup. No public dataset pairs genuine OCR screen captures
with labelled ground truth, so the screen text here is assembled from real fields
(real titles, real brands, real prices, real ratings) in the layout a product listing
would present. The *content* is real and out of domain; only the arrangement is ours.
What this measures is therefore extraction from real-world product language, not OCR
robustness.

Data source: McAuley-Lab/Amazon-Reviews-2023, raw_meta_Cell_Phones_and_Accessories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure consumer-telemetry root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.config import DATA_DIR, RANDOM_SEED
from src.ingestion.ner_extractor import POPULAR_BRANDS, extract_brand
from src.ingestion.regex_extractor import extract_price, extract_rating
from src.ingestion.pipeline import process

EXTERNAL_DIR = DATA_DIR / "external"
AMAZON_META_PATH = EXTERNAL_DIR / "amazon_meta" / "meta_shard0.parquet"
BENCHMARK_REPORT_PATH = DATA_DIR / "external" / "week1_real_data_benchmark.json"

# Amazon's metadata stores a missing price as the four-character string "None" inside a
# string column, not as a null. A null check reports full coverage and a naive cast
# silently produces nulls for two thirds of the rows, so the sentinel is handled here.
MISSING_SENTINELS = ("None", "", "null", "NaN")


def load_amazon_metadata(
    path: Optional[Union[str, Path]] = None,
    n_rows: Optional[int] = 200_000,
) -> pl.DataFrame:
    """
    Loads real product metadata and normalizes its string-typed numeric columns.

    Returns rows that have a title, a parseable price, and a rating, since a ground
    truth comparison needs all three present.
    """
    source = Path(path) if path else AMAZON_META_PATH
    if not source.exists():
        raise FileNotFoundError(
            f"Amazon metadata not found at {source}. Download it first; see "
            "week_3_real_data_validation.md for the exact command."
        )

    df = pl.read_parquet(
        source,
        columns=["title", "price", "store", "average_rating", "rating_number", "main_category"],
        n_rows=n_rows,
    )

    return (
        df.with_columns(
            [
                pl.when(pl.col("price").is_in(MISSING_SENTINELS))
                .then(None)
                .otherwise(pl.col("price"))
                .cast(pl.Float64, strict=False)
                .alias("true_price"),
                pl.col("store").str.strip_chars().alias("true_brand"),
                pl.col("average_rating").cast(pl.Float64).alias("true_rating"),
                pl.col("rating_number").cast(pl.Int64).alias("true_num_ratings"),
            ]
        )
        .filter(
            pl.col("title").is_not_null()
            & (pl.col("title").str.len_chars() > 10)
            & pl.col("true_price").is_not_null()
            & pl.col("true_rating").is_not_null()
        )
        .drop("price", "average_rating", "rating_number")
    )


def render_screen_text(row: Dict[str, Any], rng: random.Random) -> str:
    """
    Assembles a product listing string from real metadata fields.

    The layout mirrors `synthetic_data._generate_noisy_text` so the extractors face the
    same structural conventions they were built for, while every value is real.
    """
    price = float(row["true_price"])
    rating = float(row["true_rating"])
    num_ratings = int(row["true_num_ratings"] or 0)
    title = str(row["title"]).strip()

    components = [
        "Sponsored" if rng.random() < 0.25 else "",
        f"{rating:.1f}★",
        f"({num_ratings})" if num_ratings else "",
        title,
        f"${price:.2f}",
        rng.choice(["In stock", "Add to Cart", "Buy Now"]),
    ]
    return " ".join(c for c in components if c).strip()


def _brand_in_vocabulary(brand: Optional[str]) -> bool:
    """Returns whether a brand string is present in the closed matcher vocabulary."""
    if not brand:
        return False
    lowered = brand.strip().lower()
    return any(lowered == known.lower() for known in POPULAR_BRANDS)


def benchmark_field_extraction(
    df: pl.DataFrame,
    sample_size: int = 5000,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Measures price, rating, and brand extraction against real ground truth.

    Brand accuracy is reported twice: over all products, and over only those whose true
    brand exists in the matcher's closed vocabulary. The gap between the two isolates
    vocabulary coverage from matching logic.
    """
    rng = random.Random(seed)
    sample = df.sample(n=min(sample_size, df.height), seed=seed)

    stats = {
        "n": 0,
        "price_found": 0,
        "price_correct": 0,
        "rating_found": 0,
        "rating_correct": 0,
        "num_ratings_correct": 0,
        "brand_found": 0,
        "brand_correct": 0,
        "in_vocab_total": 0,
        "in_vocab_found": 0,
        "in_vocab_correct": 0,
        "brand_appears_in_title": 0,
    }
    price_errors: List[float] = []

    for row in sample.iter_rows(named=True):
        text = render_screen_text(row, rng)
        stats["n"] += 1

        price_result = extract_price(text)
        got_price = price_result.get("price")
        if got_price is not None:
            stats["price_found"] += 1
            if abs(float(got_price) - float(row["true_price"])) < 0.01:
                stats["price_correct"] += 1
            else:
                price_errors.append(abs(float(got_price) - float(row["true_price"])))

        rating_result = extract_rating(text)
        got_rating = rating_result.get("rating")
        if got_rating is not None:
            stats["rating_found"] += 1
            if abs(float(got_rating) - float(row["true_rating"])) < 0.051:
                stats["rating_correct"] += 1
        got_num = rating_result.get("num_ratings")
        if got_num is not None and int(got_num) == int(row["true_num_ratings"] or 0):
            stats["num_ratings_correct"] += 1

        true_brand = (row["true_brand"] or "").strip()
        got_brand = extract_brand(text)
        in_vocab = _brand_in_vocabulary(true_brand)
        brand_in_title = bool(true_brand) and true_brand.lower() in str(row["title"]).lower()

        if brand_in_title:
            stats["brand_appears_in_title"] += 1
        if in_vocab:
            stats["in_vocab_total"] += 1

        if got_brand:
            stats["brand_found"] += 1
            if in_vocab:
                stats["in_vocab_found"] += 1
            if true_brand and got_brand.strip().lower() == true_brand.lower():
                stats["brand_correct"] += 1
                if in_vocab:
                    stats["in_vocab_correct"] += 1

    n = max(1, stats["n"])
    in_vocab_n = max(1, stats["in_vocab_total"])
    return {
        **stats,
        "price_coverage": stats["price_found"] / n,
        "price_accuracy": stats["price_correct"] / n,
        "rating_coverage": stats["rating_found"] / n,
        "rating_accuracy": stats["rating_correct"] / n,
        "num_ratings_accuracy": stats["num_ratings_correct"] / n,
        "brand_coverage": stats["brand_found"] / n,
        "brand_accuracy": stats["brand_correct"] / n,
        "brand_in_title_share": stats["brand_appears_in_title"] / n,
        "in_vocab_share": stats["in_vocab_total"] / n,
        "in_vocab_accuracy": stats["in_vocab_correct"] / in_vocab_n,
        "mean_price_error_when_wrong": (
            sum(price_errors) / len(price_errors) if price_errors else 0.0
        ),
    }


def benchmark_full_pipeline(
    df: pl.DataFrame,
    sample_size: int = 600,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Runs the complete `process()` chain, including taxonomy and entity resolution.

    This is sampled far more thinly than the field benchmark because every call runs a
    sentence-transformer encode and a FAISS query.
    """
    rng = random.Random(seed + 1)
    sample = df.sample(n=min(sample_size, df.height), seed=seed + 1)

    categories: Dict[str, int] = {}
    confidences: List[float] = []
    canonical_ids: List[str] = []
    titles_recovered = 0
    rows = 0

    for row in sample.iter_rows(named=True):
        text = render_screen_text(row, rng)
        product = process(text, app_source="amazon_us")
        rows += 1

        label = f"{product.category_l1} > {product.category_l2}"
        categories[label] = categories.get(label, 0) + 1
        confidences.append(float(product.extraction_confidence))
        if product.canonical_product_id:
            canonical_ids.append(product.canonical_product_id)
        if product.product_title and len(product.product_title) > 3:
            titles_recovered += 1

    n = max(1, rows)
    unique_ids = len(set(canonical_ids))
    return {
        "n": rows,
        "category_distribution": dict(
            sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "distinct_categories": len(categories),
        "mean_extraction_confidence": sum(confidences) / n,
        "title_recovery_rate": titles_recovered / n,
        "canonical_ids_assigned": len(canonical_ids),
        "distinct_canonical_ids": unique_ids,
        "collision_rate": 1.0 - (unique_ids / max(1, len(canonical_ids))),
    }


def print_report(fields: Dict[str, Any], pipeline: Dict[str, Any]) -> None:
    """Prints the benchmark results as a readable report."""
    print("=" * 72)
    print("WEEK 1 EXTRACTION BENCHMARK ON REAL AMAZON PRODUCT DATA")
    print("=" * 72)

    print(f"\nDeterministic field extraction (n = {fields['n']:,})")
    print(f"  {'Field':<22} {'Coverage':>10} {'Accuracy':>10}")
    print("  " + "-" * 44)
    print(f"  {'price':<22} {fields['price_coverage']:>9.1%} {fields['price_accuracy']:>10.1%}")
    print(f"  {'rating':<22} {fields['rating_coverage']:>9.1%} {fields['rating_accuracy']:>10.1%}")
    print(f"  {'num_ratings':<22} {'-':>9} {fields['num_ratings_accuracy']:>10.1%}")
    print(f"  {'brand':<22} {fields['brand_coverage']:>9.1%} {fields['brand_accuracy']:>10.1%}")

    print(f"\nBrand extraction broken down")
    print(f"  True brand present in the closed vocabulary : {fields['in_vocab_share']:.1%} "
          f"({fields['in_vocab_total']:,} of {fields['n']:,})")
    print(f"  Accuracy restricted to those products       : {fields['in_vocab_accuracy']:.1%}")
    print(f"  True brand literally appears in the title   : {fields['brand_in_title_share']:.1%}")

    print(f"\nFull pipeline including taxonomy and resolution (n = {pipeline['n']:,})")
    print(f"  Mean extraction confidence : {pipeline['mean_extraction_confidence']:.3f}")
    print(f"  Title recovery rate        : {pipeline['title_recovery_rate']:.1%}")
    print(f"  Distinct categories used   : {pipeline['distinct_categories']}")
    print(f"  Canonical ID collision rate: {pipeline['collision_rate']:.1%}")

    print(f"\n  Assigned category distribution:")
    for label, count in list(pipeline["category_distribution"].items())[:10]:
        print(f"    {count:>5} ({count / max(1, pipeline['n']):>5.1%})  {label}")


def run_benchmark(
    field_sample: int = 5000,
    pipeline_sample: int = 600,
    n_rows: Optional[int] = 200_000,
    report_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Loads real metadata, runs both benchmarks, prints and saves the report."""
    print(f"Loading real Amazon product metadata from {AMAZON_META_PATH}...")
    df = load_amazon_metadata(n_rows=n_rows)
    print(f"  {df.height:,} products with a title, a parseable price, and a rating")

    fields = benchmark_field_extraction(df, sample_size=field_sample)
    pipeline = benchmark_full_pipeline(df, sample_size=pipeline_sample)
    print_report(fields, pipeline)

    payload = {
        "source": "McAuley-Lab/Amazon-Reviews-2023 raw_meta_Cell_Phones_and_Accessories",
        "usable_products": df.height,
        "field_extraction": fields,
        "full_pipeline": pipeline,
    }
    out = Path(report_path) if report_path else BENCHMARK_REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved benchmark report to {out}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the upstream extraction chain on real Amazon product data."
    )
    parser.add_argument("--field-sample", type=int, default=5000)
    parser.add_argument("--pipeline-sample", type=int, default=600)
    parser.add_argument("--rows", type=int, default=200_000, help="Metadata rows to scan")
    args = parser.parse_args()

    run_benchmark(
        field_sample=args.field_sample,
        pipeline_sample=args.pipeline_sample,
        n_rows=args.rows,
    )

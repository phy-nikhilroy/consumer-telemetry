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

import numpy as np
import polars as pl

from src.config import DATA_DIR, RANDOM_SEED
from src.ingestion.ner_extractor import (
    POPULAR_BRANDS,
    extract_brand_with_source,
    register_brands,
    reset_registered_brands,
)
from src.ingestion.regex_extractor import extract_price, extract_rating
from src.ingestion.pipeline import process  # noqa: E402

EXTERNAL_DIR = DATA_DIR / "external"
AMAZON_META_PATH = EXTERNAL_DIR / "amazon_meta" / "meta_shard0.parquet"
BENCHMARK_REPORT_PATH = None  # the report is written to reports/v1_2_extraction.json by merge_results

# Amazon's metadata stores a missing price as the four-character string "None" inside a
# string column, not as a null. A null check reports full coverage and a naive cast
# silently produces nulls for two thirds of the rows, so the sentinel is handled here.
MISSING_SENTINELS = ("None", "", "null", "NaN")

# A store name must appear on at least this many listings outside the evaluation sample
# to enter the brand gazetteer. One-off store names are mostly noise.
GAZETTEER_MIN_LISTINGS = 3


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


def build_store_gazetteer(
    df: pl.DataFrame,
    exclude_titles: pl.Series,
    min_listings: int = GAZETTEER_MIN_LISTINGS,
) -> List[str]:
    """
    Builds a brand gazetteer from the dataset's own store field.

    Rows whose title is in `exclude_titles` (the evaluation sample) are left out, so the
    gazetteer never contains a brand only because the row being scored contributed it.
    """
    pool = df.filter(~pl.col("title").is_in(exclude_titles.implode()))
    counts = (
        pool.filter(pl.col("true_brand").is_not_null())
        .group_by("true_brand")
        .len()
        .filter(pl.col("len") >= min_listings)
    )
    return counts["true_brand"].to_list()


def _score_brands(
    sample: pl.DataFrame,
    texts: List[str],
) -> Dict[str, Any]:
    """Scores brand extraction over a rendered sample, broken down by evidence source."""
    found = correct = 0
    by_source: Dict[str, Dict[str, int]] = {}
    for row, text in zip(sample.iter_rows(named=True), texts):
        true_brand = (row["true_brand"] or "").strip().lower()
        brand, source = extract_brand_with_source(text)
        if not brand:
            continue
        found += 1
        hit = bool(true_brand) and brand.strip().lower() == true_brand
        correct += int(hit)
        bucket = by_source.setdefault(str(source), {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(hit)

    n = max(1, len(texts))
    return {
        "coverage": found / n,
        "accuracy": correct / n,
        "precision": correct / max(1, found),
        # The share of all products given a brand that is wrong. This is the damaging
        # number: a missing brand is visible, a wrong one propagates silently.
        "wrong_share": (found - correct) / n,
        "by_source": {
            source: {**counts, "precision": counts["correct"] / max(1, counts["n"])}
            for source, counts in by_source.items()
        },
    }


def benchmark_field_extraction(
    df: pl.DataFrame,
    sample_size: int = 5000,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Measures price, rating, and brand extraction against real ground truth.

    Brand accuracy is reported over all products, over only those whose true brand is in
    the curated vocabulary (isolating matching logic from vocabulary coverage), and
    again with a store-name gazetteer built from rows outside the sample.
    """
    rng = random.Random(seed)
    sample = df.sample(n=min(sample_size, df.height), seed=seed)
    texts: List[str] = []

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
        texts.append(text)
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
        got_brand, _ = extract_brand_with_source(text)
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

    gazetteer = build_store_gazetteer(df, sample["title"])
    reset_registered_brands()
    registered = register_brands(gazetteer)
    try:
        with_gazetteer = _score_brands(sample, texts)
    finally:
        reset_registered_brands()
    without_gazetteer = _score_brands(sample, texts)

    n = max(1, stats["n"])
    in_vocab_n = max(1, stats["in_vocab_total"])
    return {
        **stats,
        "brand_open_vocabulary": without_gazetteer,
        "brand_with_store_gazetteer": {**with_gazetteer, "gazetteer_size": registered},
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


def _legacy_population_confidence(product: Any) -> float:
    """
    Recomputes the original confidence score, which counted populated fields.

    Kept only so the benchmark can compare the old and new scores on the same records.
    """
    score = 0.0
    score += 0.20 if product.brand else 0.0
    score += 0.25 if product.product_title else 0.0
    score += 0.25 if product.price is not None else 0.0
    score += 0.10 if product.currency else 0.0
    score += 0.10 if product.rating is not None else 0.0
    nearest = getattr(product, "category_l2", "Unknown")
    # The legacy classifier never abstained, so a populated category is assumed.
    score += 0.10 if nearest else 0.0
    return round(score, 4)


def _record_correctness(product: Any, row: Dict[str, Any]) -> float:
    """
    Returns the share of the three checkable fields that are right for one record.

    Brand is checked against the store field, price against the listed price, and the
    category at top level: every product here is a phone or accessory, so anything
    outside Electronics is wrong, and an abstention earns no credit.
    """
    true_brand = (row["true_brand"] or "").strip().lower()
    brand_ok = bool(product.brand) and product.brand.strip().lower() == true_brand
    price_ok = product.price is not None and abs(float(product.price) - float(row["true_price"])) < 0.01
    category_ok = product.category_l1 == "Electronics"
    return (int(brand_ok) + int(price_ok) + int(category_ok)) / 3.0


def _confidence_validity(confidences: List[float], correctness: List[float]) -> Dict[str, Any]:
    """
    Measures whether a confidence score tracks correctness.

    Reports the mean, the Spearman rank correlation with per-record correctness, and the
    mean correctness of the bottom and top confidence quintiles. A useful score has a
    positive correlation and a clear gap between quintiles; one that only counts
    populated fields has neither.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correctness, dtype=float)
    if conf.size < 5:
        return {"mean": float(conf.mean()) if conf.size else 0.0}

    if np.std(conf) == 0.0 or np.std(corr) == 0.0:
        spearman = 0.0
    else:
        conf_rank = conf.argsort(kind="stable").argsort()
        corr_rank = corr.argsort(kind="stable").argsort()
        spearman = float(np.corrcoef(conf_rank, corr_rank)[0, 1])

    order = np.argsort(conf, kind="stable")
    fifth = max(1, conf.size // 5)
    return {
        "mean": float(conf.mean()),
        "distinct_values": int(np.unique(conf).size),
        "spearman_with_correctness": spearman,
        "bottom_quintile_correctness": float(corr[order[:fifth]].mean()),
        "top_quintile_correctness": float(corr[order[-fifth:]].mean()),
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
    legacy_confidences: List[float] = []
    correctness: List[float] = []
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
        legacy_confidences.append(_legacy_population_confidence(product))
        correctness.append(_record_correctness(product, row))
        if product.canonical_product_id:
            canonical_ids.append(product.canonical_product_id)
        if product.product_title and len(product.product_title) > 3:
            titles_recovered += 1

    n = max(1, rows)
    unique_ids = len(set(canonical_ids))
    abstained = categories.get("Unknown > Unknown", 0)
    absurd = sum(
        count for label, count in categories.items()
        if not label.startswith("Electronics") and not label.startswith("Unknown")
    )
    return {
        "n": rows,
        "category_abstain_rate": abstained / n,
        "category_wrong_top_level_rate": absurd / n,
        "mean_record_correctness": float(np.mean(correctness)) if correctness else 0.0,
        "confidence_validity": {
            "evidence_weighted": _confidence_validity(confidences, correctness),
            "legacy_field_population": _confidence_validity(legacy_confidences, correctness),
        },
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


def evaluate_learned_confidence(n_rows: int = 3000) -> Dict[str, Any]:
    """
    ECE and AUC of the learned and the hand-weighted confidence on `n_rows` tune rows and
    `n_rows` held-out rows. This is the only place that names the held-out split; the tuned
    shard is excluded from it by `load_split`.
    """
    from src.config import AMAZON_SHARDS, AMAZON_TUNED_SHARD
    from src.ingestion.confidence_model import evaluate_confidence

    held_shards = [k for k in AMAZON_SHARDS if k != AMAZON_TUNED_SHARD]
    return {
        "tune": evaluate_confidence("tune", n_rows=n_rows),
        "heldout": evaluate_confidence("heldout", shards=held_shards, n_rows=n_rows),
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

    for label, key in (
        ("Open vocabulary, no gazetteer", "brand_open_vocabulary"),
        ("With store-name gazetteer", "brand_with_store_gazetteer"),
    ):
        block = fields[key]
        print(f"\n  {label}: accuracy {block['accuracy']:.1%}, coverage {block['coverage']:.1%}, "
              f"precision {block['precision']:.1%}, confidently wrong {block['wrong_share']:.1%}")
        for source, counts in sorted(block["by_source"].items(), key=lambda kv: -kv[1]["n"]):
            print(f"    {source:<22} n={counts['n']:>5,}  precision {counts['precision']:.1%}")

    print(f"\nFull pipeline including taxonomy and resolution (n = {pipeline['n']:,})")
    print(f"  Mean extraction confidence : {pipeline['mean_extraction_confidence']:.3f}")
    print(f"  Category abstain rate      : {pipeline['category_abstain_rate']:.1%}")
    print(f"  Wrong top-level category   : {pipeline['category_wrong_top_level_rate']:.1%}")
    print(f"  Mean record correctness    : {pipeline['mean_record_correctness']:.3f}")
    for label, key in (("evidence-weighted (current)", "evidence_weighted"),
                       ("field population (legacy)", "legacy_field_population")):
        v = pipeline["confidence_validity"][key]
        print(f"  Confidence, {label}: mean {v['mean']:.3f}, "
              f"Spearman with correctness {v['spearman_with_correctness']:+.3f}, "
              f"correctness bottom/top quintile {v['bottom_quintile_correctness']:.3f} / "
              f"{v['top_quintile_correctness']:.3f}")
    print(f"  Title recovery rate        : {pipeline['title_recovery_rate']:.1%}")
    print(f"  Distinct categories used   : {pipeline['distinct_categories']}")
    print(f"  Canonical ID collision rate: {pipeline['collision_rate']:.1%}")

    print(f"\n  Assigned category distribution:")
    for label, count in list(pipeline["category_distribution"].items())[:10]:
        print(f"    {count:>5} ({count / max(1, pipeline['n']):>5.1%})  {label}")


REPORT_NAME = "v1_2_extraction"
RELEASE = "1.2"
HELDOUT_ROWS_PER_SHARD = 5000
TUNE_ROWS = 5000
SYNTHETIC_SCREENS = 2000
SYNTHETIC_NOISE_RATES = (0.0, 0.01, 0.02, 0.05)
SPLIT_METRICS = (
    "brand_accuracy", "brand_confidently_wrong_rate", "price_accuracy", "rating_accuracy",
    "category_accuracy_non_abstained", "abstain_rate",
)


def score_rows(df: pl.DataFrame, seed: int = RANDOM_SEED, batch_size: int = 256) -> Dict[str, Any]:
    """
    Scores one set of prepared rows (see `confidence_model.prepare_rows`) with `process_batch`.

    Brand, price and rating accuracy are over all rows. The confidently-wrong brand rate is the
    share of rows given a brand that is wrong. Category accuracy and the abstain rate score the
    supervised model alone (`model_abstained`, `model_label` against the product-type label) on
    rows that have a product-type label; `final_category_accuracy` scores the pipeline's category
    including the prototype fallback on the same rows.
    """
    from src.ingestion.pipeline import process_batch
    from src.ingestion.pipeline import taxonomy_text
    from src.ingestion.taxonomy_classifier import classify_product

    rng = random.Random(seed)
    rows = list(df.iter_rows(named=True))
    texts = [render_screen_text(r, rng) for r in rows]
    products = process_batch(texts, app_sources=["amazon_us"] * len(texts), batch_size=batch_size)
    n = len(rows)
    brand_ok = brand_wrong = price_ok = rating_ok = 0
    cat_n = cat_abstain = cat_correct = final_correct = 0
    for row, p in zip(rows, products):
        true_brand = (row["true_brand"] or "").strip().lower()
        if p.brand:
            if true_brand and p.brand.strip().lower() == true_brand:
                brand_ok += 1
            else:
                brand_wrong += 1
        if p.price is not None and abs(float(p.price) - float(row["true_price"])) < 0.01:
            price_ok += 1
        if p.rating is not None and abs(float(p.rating) - float(row["true_rating"])) < 0.051:
            rating_ok += 1
        label = row.get("true_label")
        if not label:
            continue
        cat_n += 1
        info = classify_product(p.product_title or "", brand=p.brand,
                                model_text=taxonomy_text(p.raw_text, p.product_title))
        if info.get("model_abstained", True):
            cat_abstain += 1
        elif info.get("model_label") == label:
            cat_correct += 1
        if f"{p.category_l1} > {p.category_l2}" == label:
            final_correct += 1
    answered = cat_n - cat_abstain
    return {
        "n": n,
        "category_n": cat_n,
        "brand_accuracy": brand_ok / max(1, n),
        "brand_confidently_wrong_rate": brand_wrong / max(1, n),
        "price_accuracy": price_ok / max(1, n),
        "rating_accuracy": rating_ok / max(1, n),
        "category_accuracy_non_abstained": cat_correct / answered if answered else None,
        "abstain_rate": cat_abstain / cat_n if cat_n else None,
        "final_category_accuracy": final_correct / cat_n if cat_n else None,
    }


def load_scored_split(split: str, shards: List[str], rows_per_shard: int, seed: int = RANDOM_SEED) -> pl.DataFrame:
    """Loads `split` from each shard, prepares it, and samples at most `rows_per_shard` per shard."""
    from src.ingestion.confidence_model import prepare_rows
    from src.ingestion.real_splits import load_split

    frames = []
    for shard in shards:
        prepared = prepare_rows(load_split(split, [shard]))
        if prepared.height > rows_per_shard:
            prepared = prepared.sample(n=rows_per_shard, seed=seed)
        frames.append(prepared)
    return pl.concat(frames, how="diagonal_relaxed")


def split_gap(tune: Dict[str, Any], heldout: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Tune minus heldout for each metric in `SPLIT_METRICS` (None when either side is None)."""
    out: Dict[str, Optional[float]] = {}
    for key in SPLIT_METRICS + ("final_category_accuracy",):
        a, b = tune.get(key), heldout.get(key)
        out[key] = None if a is None or b is None else a - b
    return out


def run_real(splits: Tuple[str, ...] = ("tune", "heldout"), shards: Optional[List[str]] = None,
             seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    Scores the tune split (shard0) and/or the held-out split (the other shards).

    `shards`, when given, restricts the held-out shards; tune always uses the tuned shard.
    """
    from src.config import AMAZON_SHARDS, AMAZON_TUNED_SHARD

    held_shards = shards or [k for k in AMAZON_SHARDS if k != AMAZON_TUNED_SHARD]
    out: Dict[str, Any] = {}
    if "tune" in splits:
        out["tune"] = score_rows(load_scored_split("tune", [AMAZON_TUNED_SHARD], TUNE_ROWS, seed), seed)
        out["tune"]["shards"] = [AMAZON_TUNED_SHARD]
    if "heldout" in splits:
        out["heldout"] = score_rows(load_scored_split("heldout", held_shards, HELDOUT_ROWS_PER_SHARD, seed), seed)
        out["heldout"]["shards"] = held_shards
    if "tune" in out and "heldout" in out:
        out["gap"] = split_gap(out["tune"], out["heldout"])
    return out


def _synthetic_screen(catalog: List[dict], rng: random.Random) -> Tuple[str, List[Tuple[str, float]]]:
    """Builds one clean screen of 1 to 3 listings; returns its text and the true (brand, price) pairs."""
    from src.ingestion.screen_generator import INR_APPS, render_listing, render_screen
    from src.ingestion.synthetic_data import ACTION_SNIPPETS

    apps = sorted(ACTION_SNIPPETS)
    listings, truth = [], []
    for _ in range(rng.randint(1, 3)):
        item = catalog[rng.randrange(len(catalog))]
        app = apps[rng.randrange(len(apps))]
        action = rng.choice([a for a in ("view", "cart", "buy") if a in ACTION_SNIPPETS[app]])
        inr = app in INR_APPS
        base = float(item["base_inr"] if inr else item["base_usd"])
        price = base * rng.uniform(0.90, 1.10)
        price = float(int(round(price))) if inr else round(price, 2)
        listings.append(render_listing(item, app, action, rng.choice([0, 10, 25]), round(rng.uniform(3, 5), 1), rng, price=price))
        truth.append((item["brand"], price))
    return render_screen(listings, rng), truth


def run_synthetic(n_screens: int = SYNTHETIC_SCREENS, rates: Tuple[float, ...] = SYNTHETIC_NOISE_RATES,
                  seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    Robustness of the screen chain to OCR noise on catalog screens.

    The same clean screens are noised at each rate. `field_accuracy` is over every true
    listing (a listing counts only when the screen was segmented into the right number of
    segments and the field at its position is right). `segmentation_accuracy` is the share of
    screens with the right segment count and the right price in each segment.
    """
    from src.ingestion.catalog import build_catalog
    from src.ingestion.pipeline import process_screen
    from src.ingestion.screen_generator import apply_ocr_noise

    catalog = build_catalog()
    rng = random.Random(seed)
    screens = [_synthetic_screen(catalog, rng) for _ in range(n_screens)]
    out: Dict[str, Any] = {"n_screens": n_screens, "by_noise": {}}
    for rate in rates:
        noise_rng = random.Random(seed + int(rate * 1000))
        listings = price_ok = brand_ok = count_ok = seg_ok = 0
        for text, truth in screens:
            products = process_screen(apply_ocr_noise(text, rate, noise_rng), app_source="synthetic")
            listings += len(truth)
            right_count = len(products) == len(truth)
            count_ok += int(right_count)
            if not right_count:
                continue
            prices = [p.price is not None and abs(float(p.price) - t[1]) < 0.01 for p, t in zip(products, truth)]
            price_ok += sum(prices)
            brand_ok += sum(bool(p.brand) and p.brand.lower() == t[0].lower() for p, t in zip(products, truth))
            seg_ok += int(all(prices))
        out["by_noise"][f"{rate:.2f}"] = {
            "noise_rate": rate,
            "n_listings": listings,
            "price_accuracy": price_ok / max(1, listings),
            "brand_accuracy": brand_ok / max(1, listings),
            "segment_count_accuracy": count_ok / max(1, n_screens),
            "segmentation_accuracy": seg_ok / max(1, n_screens),
        }
    return out


def merge_results(updates: Dict[str, Any], output: Optional[Union[str, Path]] = None,
                  seed: int = RANDOM_SEED) -> Path:
    """
    Merges `updates` (top-level keys of `results`) into the report, keeping every other key,
    and writes it with the standard envelope. `output` defaults to `reports/v1_2_extraction.json`.
    """
    from src.config import REPORTS_DIR
    from src.release import reporting

    path = Path(output) if output else REPORTS_DIR / f"{REPORT_NAME}.json"
    results: Dict[str, Any] = {}
    if path.exists():
        results = dict(json.loads(path.read_text()).get("results", {}))
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(results.get(key), dict):
            results[key] = {**results[key], **value}
        else:
            results[key] = value
    original = reporting.REPORTS_DIR
    reporting.REPORTS_DIR = path.parent
    try:
        return reporting.write_report(path.stem, RELEASE, results, seed=seed)
    finally:
        reporting.REPORTS_DIR = original


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: runs the real and synthetic benchmarks and merges them into the report."""
    parser = argparse.ArgumentParser(description="Held-out benchmark of the extraction chain.")
    parser.add_argument("--split", choices=("tune", "heldout"), default=None,
                        help="Score only this real split (default both)")
    parser.add_argument("--shards", nargs="+", default=None, help="Held-out shard keys")
    parser.add_argument("--output", default=None, help="Report path (default reports/v1_2_extraction.json)")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-confidence", action="store_true")
    args = parser.parse_args(argv)

    splits = (args.split,) if args.split else ("tune", "heldout")
    updates: Dict[str, Any] = {"real": run_real(splits, args.shards)}
    if not args.skip_synthetic:
        updates["synthetic"] = run_synthetic()
    if not args.skip_confidence and not args.split:
        updates["confidence"] = evaluate_learned_confidence()
    print(json.dumps(updates, indent=2, default=str))
    print(merge_results(updates, args.output))


if __name__ == "__main__":
    main()

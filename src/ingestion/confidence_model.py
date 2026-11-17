"""Learned extraction confidence: a logistic regression over the evidence behind a record.

Each rendered real listing is run through ``process()``. The label is 1 when the brand
matches the Amazon ``store`` (case-insensitive, as ``real_data_benchmark._score_brands``
scores it), the price is within 1 percent of the listed price, and the category label
(``"<l1> > <l2>"`` from ``taxonomy_train.product_type_label``) matches. The model maps ten
evidence features to the probability that all three are right. The artifact is committed
at ``config.CONFIDENCE_ARTIFACT_PATH``; ``pipeline.calculate_extraction_confidence`` uses
it when present and otherwise falls back to the hand-weighted score.

This module never names the held-out split; evaluation takes the split name as an argument
and ``real_data_benchmark.py`` is the caller that passes it.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import polars as pl

from src.config import (
    AMAZON_SHARDS,
    CONFIDENCE_ARTIFACT_PATH,
    CONFIDENCE_ECE_BINS,
    CONFIDENCE_EVAL_ROWS,
    CONFIDENCE_PRICE_TOLERANCE,
    CONFIDENCE_TRAIN_ROWS,
    RANDOM_SEED,
)

BRAND_SOURCE_NAMES = ("leading_match", "leading_token", "match")

FEATURE_NAMES = (
    "brand_leading_match",
    "brand_leading_token",
    "brand_mid_match",
    "price_present",
    "price_verified",
    "currency_present",
    "rating_present",
    "category_probability",
    "category_margin",
    "title_token_count",
)

_MODEL_CACHE: Dict[str, Any] = {}


def _brand_source_flags(source: Optional[str]) -> List[float]:
    """One-hot of the brand source in the order of the first three features."""
    from src.ingestion.ner_extractor import (
        BRAND_SOURCE_LEADING_MATCH,
        BRAND_SOURCE_LEADING_TOKEN,
        BRAND_SOURCE_MATCH,
    )

    return [
        float(source == BRAND_SOURCE_LEADING_MATCH),
        float(source == BRAND_SOURCE_LEADING_TOKEN),
        float(source == BRAND_SOURCE_MATCH),
    ]


def feature_vector(extracted: Mapping[str, Any]) -> List[float]:
    """
    The ten evidence features of one extracted record, in `FEATURE_NAMES` order.

    Reads `brand`, `brand_source`, `price`, `price_verified`, `currency`, `rating`,
    `category_probability` (falls back to `category_similarity` when absent),
    `category_model_margin`, `category_abstained` (the supervised model's own
    outputs, whichever path answered) and `product_title`. A brand source counts only when a brand is present.
    """
    source = extracted.get("brand_source") if extracted.get("brand") else None
    title = extracted.get("product_title") or ""
    category = extracted.get("category_l2")
    abstained = bool(extracted.get("category_abstained")) or category == "Unknown"
    probability = extracted.get("category_probability")
    if probability is None:
        probability = extracted.get("category_similarity")
    probability = 0.0 if abstained else float(probability or 0.0)
    margin = 0.0 if abstained else float(extracted.get("category_model_margin") or 0.0)
    return [
        *_brand_source_flags(source),
        float(extracted.get("price") is not None),
        float(bool(extracted.get("price_verified")) and extracted.get("price") is not None),
        float(bool(extracted.get("currency"))),
        float(extracted.get("rating") is not None),
        probability,
        margin,
        float(len(str(title).split())),
    ]


def load_model(path: Optional[Path] = None) -> Optional[Any]:
    """Loads the artifact's model, cached by path and mtime; None when the file is absent."""
    import joblib

    target = Path(path) if path is not None else CONFIDENCE_ARTIFACT_PATH
    if not target.exists():
        return None
    key = f"{target}:{target.stat().st_mtime_ns}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = joblib.load(target)["model"]
    return _MODEL_CACHE[key]


def predict_confidence(extracted: Mapping[str, Any], path: Optional[Path] = None) -> Optional[float]:
    """Probability from the learned model in [0, 1], or None when no artifact exists."""
    model = load_model(path)
    if model is None:
        return None
    p = float(model.predict_proba(np.asarray([feature_vector(extracted)], dtype=float))[0, 1])
    return round(min(1.0, max(0.0, p)), 4)


def prepare_rows(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds `true_price`, `true_brand`, `true_rating`, `true_num_ratings` and `true_label` to raw
    shard rows and keeps those with a parseable price and a rating. `true_label` is null when
    the category path has no product type.
    """
    from src.ingestion.real_data_benchmark import MISSING_SENTINELS
    from src.ingestion.taxonomy_train import product_type_label

    labels = []
    for path in df["categories"].to_list():
        pair = product_type_label(path)
        labels.append(f"{pair[0]} > {pair[1]}" if pair else None)
    return (
        df.with_columns(
            pl.when(pl.col("price").is_in(MISSING_SENTINELS))
            .then(None)
            .otherwise(pl.col("price"))
            .cast(pl.Float64, strict=False)
            .alias("true_price"),
            pl.col("store").str.strip_chars().alias("true_brand"),
            pl.col("average_rating").cast(pl.Float64).alias("true_rating"),
            pl.col("rating_number").cast(pl.Int64).alias("true_num_ratings"),
            pl.Series("true_label", labels, dtype=pl.String),
        )
        .filter(
            (pl.col("title").str.len_chars() > 10)
            & pl.col("true_price").is_not_null()
            & (pl.col("true_price") > 0)
            & pl.col("true_rating").is_not_null()
        )
    )


def record_label(extracted: Mapping[str, Any], row: Mapping[str, Any]) -> int:
    """1 when brand, price (within 1 percent) and category label all match the row, else 0."""
    true_brand = (row.get("true_brand") or "").strip().lower()
    brand = (extracted.get("brand") or "").strip().lower()
    brand_ok = bool(true_brand) and brand == true_brand
    price = extracted.get("price")
    true_price = float(row["true_price"])
    price_ok = price is not None and abs(float(price) - true_price) <= CONFIDENCE_PRICE_TOLERANCE * true_price
    category_ok = row.get("true_label") is not None and extracted.get("full_category") == row["true_label"]
    return int(brand_ok and price_ok and category_ok)


def build_dataset(rows: pl.DataFrame, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Renders and processes each prepared row. Returns the feature matrix, the labels, and the
    hand-weighted score of every record.
    """
    from src.ingestion.pipeline import calculate_hand_weighted_confidence, process, product_to_evidence
    from src.ingestion.real_data_benchmark import render_screen_text

    rng = random.Random(seed)
    xs: List[List[float]] = []
    ys: List[int] = []
    hand: List[float] = []
    for row in rows.iter_rows(named=True):
        product = process(render_screen_text(row, rng), app_source="amazon_us")
        evidence = product_to_evidence(product)
        xs.append(feature_vector(evidence))
        ys.append(record_label(evidence, row))
        hand.append(calculate_hand_weighted_confidence(evidence))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int), hand  # type: ignore[return-value]


def expected_calibration_error(probs: Sequence[float], labels: Sequence[int], bins: int = CONFIDENCE_ECE_BINS) -> float:
    """Expected calibration error over `bins` equal-width bins on [0, 1]."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return 0.0
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def auc_score(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """AUC of scores against binary labels; None when only one class is present."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def _load_prepared(split: str, shards: Sequence[str], n_rows: int, seed: int) -> pl.DataFrame:
    """Loads, prepares and samples rows of one split."""
    from src.ingestion.real_splits import load_split

    return _sample(prepare_rows(load_split(split, shards)), n_rows, seed)


def _sample(df: pl.DataFrame, n_rows: int, seed: int) -> pl.DataFrame:
    """Seeded sample of at most n_rows rows."""
    return df.sample(n=n_rows, seed=seed) if df.height > n_rows else df


def train_confidence(
    output_path: Path = CONFIDENCE_ARTIFACT_PATH,
    shards: Optional[Sequence[str]] = None,
    n_rows: int = CONFIDENCE_TRAIN_ROWS,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Fits the logistic regression on `n_rows` train-split rows and writes the artifact."""
    import joblib
    from sklearn.linear_model import LogisticRegression

    shards = list(shards) if shards is not None else list(AMAZON_SHARDS)
    rows = _load_prepared("train", shards, n_rows, seed)
    x, y, _ = build_dataset(rows, seed)
    model = LogisticRegression(max_iter=2000).fit(x, y)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(FEATURE_NAMES), "trained_on_rows": int(len(y))}, output_path)
    _MODEL_CACHE.clear()
    return {"trained_on_rows": int(len(y)), "positive_rate": float(y.mean()), "output": str(output_path)}


def evaluate_confidence(
    split: str,
    shards: Optional[Sequence[str]] = None,
    n_rows: int = CONFIDENCE_EVAL_ROWS,
    seed: int = RANDOM_SEED,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    ECE (10 equal-width bins) and AUC against the label, for the learned and the
    hand-weighted score, on `n_rows` rows of the named split. The caller supplies the
    split name and the shards it may use.
    """
    model = load_model(path)
    if model is None:
        raise FileNotFoundError("no confidence artifact; run train_confidence first")
    shards = list(shards) if shards is not None else list(AMAZON_SHARDS)
    rows = _load_prepared(split, shards, n_rows, seed)
    x, y, hand = build_dataset(rows, seed)
    learned = model.predict_proba(x)[:, 1]
    return {
        "split": split,
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "learned": {"ece": expected_calibration_error(learned, y), "auc": auc_score(learned, y)},
        "hand_weighted": {"ece": expected_calibration_error(hand, y), "auc": auc_score(hand, y)},
    }


def main(argv: Optional[List[str]] = None) -> None:
    """Trains the confidence model, then prints tune-split metrics."""
    parser = argparse.ArgumentParser(description="Train the learned extraction confidence model.")
    parser.add_argument("--output", type=Path, default=CONFIDENCE_ARTIFACT_PATH)
    parser.add_argument("--rows", type=int, default=CONFIDENCE_TRAIN_ROWS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    print(train_confidence(args.output, n_rows=args.rows, seed=args.seed))
    print(evaluate_confidence("tune", path=args.output, seed=args.seed))


if __name__ == "__main__":
    main()

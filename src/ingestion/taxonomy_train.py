"""
Trains the supervised product taxonomy classifier with calibrated abstention.

The v1.0 classifier matched titles against hand-written category descriptions that had been
tuned while looking at the evaluation listings. This module learns categories instead, from
the Amazon `categories` list of real listings in the train split (`real_splits.load_split`),
and picks its abstention threshold on the tune split, so the held-out shards are never seen.

Labels are `"<categories[0]> > <product node>"`, where the product node is the first later
node that is not an audience (Women, Men, ...); the plan's plain second level labelled most
fashion by audience and some phones by a merchandising widget. A label is kept only if it has at least
`TAXONOMY_MIN_LABEL_ROWS` train rows. Titles are encoded with the sentence encoder,
L2-normalised, and fed to a class-balanced logistic regression. The abstention threshold
`tau` is the smallest value of `TAXONOMY_TAU_GRID` at which tune rows predicted with
`max_prob >= tau` reach `TAXONOMY_TARGET_ACCURACY`; tune rows whose true label is not in the
label set count as wrong, so the threshold also guards against unfamiliar products.

Run: .venv/bin/python -m src.ingestion.taxonomy_train
writes src/ingestion/artifacts/taxonomy_lr.joblib.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.config import (  # noqa: E402
    AMAZON_SHARDS,
    RANDOM_SEED,
    TAXONOMY_ARTIFACT_PATH,
    TAXONOMY_AUDIENCE_NODES,
    TAXONOMY_ENCODER_NAME,
    TAXONOMY_MAX_ROWS_PER_LABEL,
    TAXONOMY_MAX_TRAIN_ROWS,
    TAXONOMY_MIN_LABEL_ROWS,
    TAXONOMY_NON_PRODUCT_NODES,
    TAXONOMY_TARGET_ACCURACY,
    TAXONOMY_TAU_FALLBACK,
    TAXONOMY_TAU_GRID,
    TAXONOMY_TUNE_ROWS,
)

LABEL_SEPARATOR = " > "


def product_type_label(path: Optional[Sequence[str]]) -> Optional[Tuple[str, str]]:
    """
    The (category_l1, category_l2) of a category path: the first level, and the first later
    node that is not an audience node. None when the path has fewer than two levels, when only
    audience nodes follow the first level, or when the product node is a merchandising node.
    """
    if path is None or len(path) < 2:
        return None
    rest = [node for node in list(path)[1:] if node not in TAXONOMY_AUDIENCE_NODES]
    if not rest or rest[0] in TAXONOMY_NON_PRODUCT_NODES:
        return None
    return str(path[0]), str(rest[0])


def build_label_frame(df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns `title`, `category_l1`, `category_l2`, and `label` (`"l1 > l2"`, see
    `product_type_label`) for rows with a non-empty title and a usable category path.
    """
    base = df.select("title", "categories").filter(
        pl.col("title").is_not_null() & (pl.col("title").str.strip_chars() != "")
    )
    pairs = [product_type_label(path) for path in base["categories"].to_list()]
    out = base.drop("categories").with_columns(
        pl.Series("category_l1", [p[0] if p else None for p in pairs], dtype=pl.String),
        pl.Series("category_l2", [p[1] if p else None for p in pairs], dtype=pl.String),
    ).drop_nulls("category_l1")
    return out.with_columns((pl.col("category_l1") + LABEL_SEPARATOR + pl.col("category_l2")).alias("label"))


def select_labels(labelled: pl.DataFrame, min_rows: int = TAXONOMY_MIN_LABEL_ROWS) -> List[str]:
    """Labels with at least `min_rows` rows, sorted by name."""
    counts = labelled.group_by("label").len()
    return sorted(counts.filter(pl.col("len") >= min_rows)["label"].to_list())


def sample_training_rows(
    labelled: pl.DataFrame,
    labels: Sequence[str],
    per_label: int = TAXONOMY_MAX_ROWS_PER_LABEL,
    total: int = TAXONOMY_MAX_TRAIN_ROWS,
    seed: int = RANDOM_SEED,
) -> pl.DataFrame:
    """
    Keeps rows of the kept labels, at most `per_label` per label and at most `total`
    overall, sampling with `seed`. Rows of other labels are dropped.
    """
    kept = labelled.filter(pl.col("label").is_in(list(labels)))
    parts = []
    for i, label in enumerate(sorted(labels)):
        rows = kept.filter(pl.col("label") == label)
        if rows.height > per_label:
            rows = rows.sample(n=per_label, seed=seed + i)
        parts.append(rows)
    out = pl.concat(parts) if parts else kept.head(0)
    if out.height > total:
        out = out.sample(n=total, seed=seed)
    return out.sort("label", "title")


def choose_tau(
    max_prob: np.ndarray,
    correct: np.ndarray,
    grid: Sequence[float] = TAXONOMY_TAU_GRID,
    target: float = TAXONOMY_TARGET_ACCURACY,
    fallback: float = TAXONOMY_TAU_FALLBACK,
) -> float:
    """
    The smallest grid value at which rows with `max_prob >= tau` are at least `target`
    accurate. Grid values that leave no confident row do not qualify. Returns `fallback`
    when none qualifies.
    """
    max_prob = np.asarray(max_prob, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    for tau in sorted(grid):
        confident = max_prob >= tau
        if confident.any() and correct[confident].mean() >= target:
            return float(tau)
    return float(fallback)


def load_encoder(name: str = TAXONOMY_ENCODER_NAME) -> Any:
    """Loads the sentence encoder."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def encode_titles(encoder: Any, titles: Sequence[str], batch_size: int = 256) -> np.ndarray:
    """L2-normalised title embeddings as a float32 matrix."""
    vectors = encoder.encode(list(titles), batch_size=batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def fit_classifier(x: np.ndarray, y: Sequence[str]) -> Any:
    """Class-balanced logistic regression on the embeddings."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    model.fit(x, np.asarray(y))
    return model


def train_taxonomy(
    shards: Optional[Sequence[str]] = None,
    output_path: Path = TAXONOMY_ARTIFACT_PATH,
    seed: int = RANDOM_SEED,
    encoder: Any = None,
    tune_rows: int = TAXONOMY_TUNE_ROWS,
) -> Dict[str, Any]:
    """
    Trains the classifier on the train split of `shards` (default: every shard in
    `AMAZON_SHARDS`), chooses `tau` on the tune split, and writes the artifact
    `{"model", "labels", "tau", "encoder_name", "trained_on_rows"}` to `output_path`.
    Returns a summary with the label count, row counts, tau, and tune accuracy.
    """
    import joblib

    from src.ingestion.real_splits import load_split

    shards = list(shards) if shards is not None else list(AMAZON_SHARDS)
    encoder = encoder if encoder is not None else load_encoder()
    started = time.time()

    train_labelled = build_label_frame(load_split("train", shards))
    labels = select_labels(train_labelled)
    if len(labels) < 2:
        raise ValueError(f"need at least two labels with {TAXONOMY_MIN_LABEL_ROWS} rows, found {labels}")
    train_rows = sample_training_rows(train_labelled, labels, seed=seed)
    model = fit_classifier(encode_titles(encoder, train_rows["title"].to_list()), train_rows["label"].to_list())

    tune_labelled = build_label_frame(load_split("tune", shards))
    if tune_labelled.height > tune_rows:
        tune_labelled = tune_labelled.sample(n=tune_rows, seed=seed)
    probs = model.predict_proba(encode_titles(encoder, tune_labelled["title"].to_list()))
    predicted = model.classes_[probs.argmax(axis=1)]
    correct = predicted == tune_labelled["label"].to_numpy()
    max_prob = probs.max(axis=1)
    tau = choose_tau(max_prob, correct)

    artifact = {
        "model": model,
        "labels": [str(c) for c in model.classes_],
        "tau": tau,
        "encoder_name": TAXONOMY_ENCODER_NAME,
        "trained_on_rows": int(train_rows.height),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    confident = max_prob >= tau
    return {
        "labels": len(labels),
        "trained_on_rows": int(train_rows.height),
        "tune_rows": int(tune_labelled.height),
        "tau": tau,
        "tune_accuracy_all": round(float(correct.mean()), 4),
        "tune_accuracy_confident": round(float(correct[confident].mean()), 4) if confident.any() else None,
        "tune_abstain_rate": round(float(1.0 - confident.mean()), 4),
        "seconds": round(time.time() - started, 1),
        "output": str(output_path),
    }


def main(argv: Optional[List[str]] = None) -> None:
    """Trains the taxonomy classifier and prints its summary."""
    parser = argparse.ArgumentParser(description="Train the supervised taxonomy classifier.")
    parser.add_argument("--shards", nargs="+", default=None, help="keys of AMAZON_SHARDS (default: all)")
    parser.add_argument("--output", type=Path, default=TAXONOMY_ARTIFACT_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    summary = train_taxonomy(args.shards, args.output, args.seed)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

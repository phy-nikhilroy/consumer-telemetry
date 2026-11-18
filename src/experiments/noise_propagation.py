"""Extraction-error propagation: how OCR noise in simulated screens reaches event accuracy and the propensity model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PROJECT_ROOT, RANDOM_SEED, USD_TO_INR_RATE

DEFAULT_OUT = PROJECT_ROOT / "data" / "experiments" / "v1_3_noise"
NOISE_LEVELS = (0.0, 0.02, 0.05)
PRICE_TOLERANCE = 0.01
LOAD_WORKERS = 4


def _to_inr(price: pl.Expr, currency: pl.Expr) -> pl.Expr:
    """Expression converting a price to INR by its currency code."""
    return pl.when(currency == "USD").then(price * USD_TO_INR_RATE).otherwise(price)


def event_accuracy(db_path: Path, truth_parquet: Path) -> Dict[str, Any]:
    """Event-level accuracy of the warehouse against the simulator truth, matched on event_id.

    Action is exact. Price is within 1% in INR. Identity is correct when the event's
    canonical_product_id equals the modal id of its true SKU. Truth events missing from the
    warehouse count as incorrect on every field.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        facts = pl.from_arrow(con.execute(
            "SELECT event_id, action, price, currency, canonical_product_id FROM fact_events"
        ).to_arrow_table())
    finally:
        con.close()
    truth = pl.read_parquet(truth_parquet)
    joined = truth.join(facts, on="event_id", how="left", suffix="_w")
    n = joined.height
    if n == 0:
        nan = float("nan")
        return {"events": 0, "matched": 0, "action": nan, "price": nan, "identity": nan, "sku_collisions": 0}

    modal = (
        joined.filter(pl.col("canonical_product_id").is_not_null())
        .group_by("true_sku_key", "canonical_product_id").len()
        .sort(["true_sku_key", "len", "canonical_product_id"], descending=[False, True, False])
        .group_by("true_sku_key", maintain_order=True).first()
        .select("true_sku_key", pl.col("canonical_product_id").alias("modal_id"))
    )
    joined = joined.join(modal, on="true_sku_key", how="left")
    t_inr = _to_inr(pl.col("true_price"), pl.col("true_currency"))
    w_inr = _to_inr(pl.col("price"), pl.col("currency"))
    scored = joined.with_columns(
        (pl.col("action") == pl.col("true_action")).fill_null(False).alias("ok_action"),
        pl.when(pl.col("true_price").is_null() & pl.col("price").is_null()).then(True)
        .when(pl.col("true_price").is_null() | pl.col("price").is_null()).then(False)
        .otherwise((w_inr - t_inr).abs() <= PRICE_TOLERANCE * t_inr.abs()).alias("ok_price"),
        (pl.col("canonical_product_id") == pl.col("modal_id")).fill_null(False).alias("ok_identity"),
    )
    return {
        "events": n,
        "matched": int(joined["canonical_product_id"].is_not_null().sum()),
        "action": float(scored["ok_action"].mean()),
        "price": float(scored["ok_price"].mean()),
        "identity": float(scored["ok_identity"].mean()),
        "sku_collisions": int(modal.height and (
            joined.filter(pl.col("canonical_product_id").is_not_null())
            .group_by("true_sku_key").agg(pl.col("canonical_product_id").n_unique().alias("k"))
            .filter(pl.col("k") > 1).height)),
    }


def run_level(noise: float, out_dir: Path, users: int, days: int, seed: int) -> Dict[str, Any]:
    """Runs simulate, load, sessionize, features, validation and propensity for one noise level under `out_dir`.

    Returns the accuracy block, the propensity metrics, and per-user test scores for pairing.
    """
    from src.experiments.run_seeds import _deterministic_lightgbm
    from src.features.feature_store import generate_train_test_datasets, get_modeling_matrices
    from src.features.feature_validation import run_feature_validation_pipeline
    from src.features.sessionizer import run_sessionization_pipeline
    from src.ingestion.simulate_users import run_simulation
    from src.models.propensity import PRIMARY_TARGET, run_training_pipeline

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "warehouse.duckdb"
    features_dir = out_dir / "features"
    artifacts_dir = out_dir / "artifacts"
    events_path = out_dir / "events.parquet"
    truth_path = out_dir / "truth.parquet"
    with _deterministic_lightgbm():
        sim = run_simulation(num_users=users, days=days, db_path=str(db_path), output_parquet=str(events_path),
                             seed=seed, ocr_noise=noise, workers=LOAD_WORKERS, truth_parquet=str(truth_path))
        run_sessionization_pipeline(db_path=db_path)
        generate_train_test_datasets(db_path=db_path, features_dir=features_dir)
        run_feature_validation_pipeline(features_dir=features_dir)
        prop = run_training_pipeline(features_dir=features_dir, artifacts_dir=artifacts_dir)
        X_test, y_test, _ = get_modeling_matrices(split="test", target_col=PRIMARY_TARGET, features_dir=features_dir)
        scores = prop["model"].predict_proba(X_test)[:, 1]
    test_users = pl.read_parquet(features_dir / "test_features.parquet", columns=["user_id"])["user_id"].to_numpy()
    metrics = prop["metrics_path"]
    import json
    test_block = json.loads(Path(metrics).read_text(encoding="utf-8"))["test"]
    return {
        "noise": noise,
        "total_events": sim["total_events"],
        "accuracy": event_accuracy(db_path, truth_path),
        "propensity_test": {
            "roc_auc": test_block["roc_auc"], "ci": test_block["ci"]["roc_auc"],
            "pr_auc": test_block.get("pr_auc"),
        },
        "_scores": {"users": test_users, "y": np.asarray(y_test), "p": scores},
    }


def paired_delta(level: Dict[str, Any], base: Dict[str, Any], n_boot: int = 2000) -> Dict[str, Any]:
    """Paired bootstrap ROC-AUC delta (level minus base) over the test users both runs share."""
    from src.models.baselines import paired_bootstrap_delta

    a, b = level["_scores"], base["_scores"]
    shared = sorted(set(a["users"].tolist()) & set(b["users"].tolist()))
    ia = {u: i for i, u in reversed(list(enumerate(a["users"].tolist())))}
    ib = {u: i for i, u in reversed(list(enumerate(b["users"].tolist())))}
    sa = np.array([ia[u] for u in shared], dtype=int)
    sb = np.array([ib[u] for u in shared], dtype=int)
    if len(shared) == 0:
        nan = float("nan")
        return {"delta": nan, "lo": nan, "hi": nan, "shared_users": 0}
    y = b["y"][sb]
    out = paired_bootstrap_delta(y, a["p"][sa], b["p"][sb], n_boot=n_boot)
    out["shared_users"] = len(shared)
    return out


def run_noise_propagation(
    out_root: Path = DEFAULT_OUT,
    noise_levels: Sequence[float] = NOISE_LEVELS,
    users: int = 400,
    days: int = 30,
    seed: int = RANDOM_SEED,
    n_boot: int = 2000,
) -> Dict[str, Any]:
    """Runs every noise level in `out_root/<level>/` and reports accuracy, ROC-AUC with CI and paired deltas.

    The first level is the baseline for the paired deltas.
    """
    runs: Dict[float, Dict[str, Any]] = {}
    for noise in noise_levels:
        runs[noise] = run_level(noise, Path(out_root) / f"{noise:g}", users, days, seed)
    base = runs[noise_levels[0]]
    levels: Dict[str, Any] = {}
    for noise, run in runs.items():
        entry = {k: v for k, v in run.items() if k != "_scores"}
        entry["delta_vs_baseline"] = paired_delta(run, base, n_boot=n_boot)
        levels[f"{noise:g}"] = entry
    return {"seed": seed, "users": users, "days": days, "baseline_noise": noise_levels[0],
            "noise_levels": [float(x) for x in noise_levels], "levels": levels}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure how OCR extraction noise reaches the propensity model.")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Root for per-level isolated outputs")
    parser.add_argument("--noise", type=float, nargs="+", default=list(NOISE_LEVELS))
    parser.add_argument("--users", type=int, default=400)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    from src.release.reporting import write_report

    results = run_noise_propagation(Path(args.out), args.noise, args.users, args.days, args.seed, args.n_boot)
    print(f"Wrote {write_report('v1_3_noise_propagation', '1.3', results, seed=args.seed)}")

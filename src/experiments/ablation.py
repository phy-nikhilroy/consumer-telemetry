"""Feature-block ablation (S6): models A (rfm), B (+funnel), C (+cross_app), D (all blocks).

For each simulation seed the warehouse is built under data/experiments, then LightGBM models
are fitted on the nested block sets for purchase propensity and `switched_given_purchase`.
Test ROC-AUC with a bootstrap CI and paired bootstrap deltas B-A, C-A, D-A are aggregated
across seeds and written to reports/v1_4_ablation.json. A RetailRocket section runs A, B and
D-without-cross_app on the v1.1 dates. Nothing is written outside the chosen output directory
except the report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from src.config import (
    CONDITIONAL_BOOTSTRAP_SAMPLES,
    FEATURE_BLOCKS,
    PRIMARY_TARGET,
    PROJECT_ROOT,
    SWITCHING_CONDITIONAL_TARGET,
    SWITCHING_TARGET,
)
from src.features.feature_store import fill_missing_features
from src.models.baselines import bootstrap_metric_ci, paired_bootstrap_delta

DEFAULT_OUT = PROJECT_ROOT / "data" / "experiments" / "v1_4_ablation"
REPORT_NAME = "v1_4_ablation"
RELEASE = "1.4"
TARGETS = (PRIMARY_TARGET, SWITCHING_CONDITIONAL_TARGET)
CAVEAT = (
    "On simulated data this measures recovery of a built-in mechanism: the simulator draws "
    "behaviour from archetype parameters, so a block that helps shows the model recovers a "
    "mechanism the generator put there, not a property of real consumers. The RetailRocket "
    "section is the real-data check."
)


def model_feature_sets(include_cross_app_in_d: bool = True) -> Dict[str, List[str]]:
    """Returns the feature list of models A, B, C and D (D optionally without cross_app)."""
    rfm = list(FEATURE_BLOCKS["rfm"])
    d_blocks = [b for b in FEATURE_BLOCKS if include_cross_app_in_d or b != "cross_app"]
    return {
        "A": rfm,
        "B": rfm + list(FEATURE_BLOCKS["funnel"]),
        "C": rfm + list(FEATURE_BLOCKS["cross_app"]),
        "D": [f for b in d_blocks for f in FEATURE_BLOCKS[b]],
    }


def _matrix(df: pl.DataFrame, names: Sequence[str]) -> np.ndarray:
    """Model input matrix with the project's missing-value semantics."""
    return fill_missing_features(df, list(names)).to_numpy().astype(np.float32)


def target_frames(train: pl.DataFrame, test: pl.DataFrame, target: str):
    """Returns (train, test) restricted and labelled for `target` (buyers only for the conditional)."""
    if target == SWITCHING_CONDITIONAL_TARGET:
        from src.models.switching import build_conditional_target

        return build_conditional_target(train), build_conditional_target(test)
    return train, test


def evaluate_target(
    train: pl.DataFrame,
    test: pl.DataFrame,
    target: str,
    feature_sets: Dict[str, List[str]],
    n_boot: int = CONDITIONAL_BOOTSTRAP_SAMPLES,
) -> Dict[str, Any]:
    """Fits each named model on `train` and scores `test`: AUC with CI and paired deltas vs A."""
    from src.models.propensity import train_propensity_model

    tr, te = target_frames(train, test, target)
    y_tr = tr[target].to_numpy()
    y_te = te[target].to_numpy()
    groups = tr["user_id"].to_numpy()
    if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
        raise ValueError(f"target {target} has a single class in train or test")
    scores: Dict[str, np.ndarray] = {}
    out: Dict[str, Any] = {"n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                           "test_positive_rate": float(np.mean(y_te)), "models": {}, "deltas": {}}
    for name, feats in feature_sets.items():
        feats = [f for f in feats if f in tr.columns]
        model = train_propensity_model(_matrix(tr, feats), y_tr, groups=groups)
        scores[name] = model.predict_proba(_matrix(te, feats))[:, 1]
        ci = bootstrap_metric_ci(y_te, scores[name], roc_auc_score, n_boot=n_boot)
        out["models"][name] = {"n_features": len(feats), "roc_auc": ci["point"],
                               "ci_lo": ci["lo"], "ci_hi": ci["hi"]}
    if "A" in scores:
        for new in scores:
            if new != "A":
                out["deltas"][f"{new}-A"] = paired_bootstrap_delta(
                    y_te, scores[new], scores["A"], roc_auc_score, n_boot=n_boot)
    return out


def aggregate_seeds(per_seed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates per-seed results to mean, std, min, max, n of each AUC and delta, per target."""
    from src.experiments.run_seeds import _aggregate

    agg: Dict[str, Any] = {}
    for target in TARGETS:
        runs = [r[target] for r in per_seed.values() if target in r]
        models: Dict[str, Any] = {}
        for name in sorted({m for r in runs for m in r["models"]}):
            models[name] = _aggregate([r["models"][name]["roc_auc"] for r in runs if name in r["models"]])
        deltas: Dict[str, Any] = {}
        for key in sorted({k for r in runs for k in r["deltas"]}):
            vals = [r["deltas"][key] for r in runs if key in r["deltas"]]
            deltas[key] = {
                **_aggregate([v["delta"] for v in vals]),
                "seeds_ci_excludes_zero": int(sum(1 for v in vals if v["lo"] > 0 or v["hi"] < 0)),
            }
        agg[target] = {"roc_auc": models, "deltas": deltas}
    return agg


def run_one_seed(seed: int, out_dir: Path, users: int, days: int, n_boot: int) -> Dict[str, Any]:
    """Simulates one seed under `out_dir`, builds features and evaluates both targets."""
    from src.experiments.run_seeds import _deterministic_lightgbm
    from src.features.feature_store import generate_train_test_datasets, split_for_warehouse
    from src.features.sessionizer import run_sessionization_pipeline
    from src.ingestion.simulate_users import run_simulation

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "warehouse.duckdb"
    features_dir = out_dir / "features"
    run_simulation(num_users=users, days=days, db_path=str(db_path),
                   output_parquet=str(out_dir / "events.parquet"), seed=seed)
    run_sessionization_pipeline(db_path=db_path)
    train_refs, test_ref = split_for_warehouse(db_path)
    train, test = generate_train_test_datasets(train_ref_dates=train_refs, test_ref_date=test_ref,
                                               db_path=db_path, features_dir=features_dir)
    sets = model_feature_sets()
    with _deterministic_lightgbm():
        return {t: evaluate_target(train, test, t, sets, n_boot) for t in TARGETS}


def run_retailrocket(out_dir: Path, n_boot: int) -> Dict[str, Any]:
    """Runs A, B and D-without-cross_app on RetailRocket with the v1.1 dates (propensity target)."""
    import duckdb

    from src.experiments.retailrocket_protocol import (
        RR_DB_PATH, RR_TEST_REF, RR_TRAIN_REFS, ensure_current_schema, resolve_dates)
    from src.experiments.run_seeds import _deterministic_lightgbm
    from src.features.feature_store import generate_train_test_datasets

    ensure_current_schema(RR_DB_PATH)
    con = duckdb.connect(str(RR_DB_PATH), read_only=True)
    try:
        hi = con.execute("SELECT MAX(timestamp) FROM fact_events").fetchone()[0]
    finally:
        con.close()
    trains, test_ref, shift = resolve_dates(RR_TRAIN_REFS, RR_TEST_REF, hi)
    train, test = generate_train_test_datasets(trains, test_ref, db_path=RR_DB_PATH,
                                               features_dir=Path(out_dir) / "retailrocket_features")
    full = model_feature_sets(include_cross_app_in_d=False)
    sets = {"A": full["A"], "B": full["B"], "D_without_cross_app": full["D"]}
    with _deterministic_lightgbm():
        res = evaluate_target(train, test, PRIMARY_TARGET, sets, n_boot)
    return {"dates": {"train_refs": trains, "test_ref": test_ref, "weeks_shifted": shift},
            "target": PRIMARY_TARGET, **res}


def run_ablation(seeds: Sequence[int], out_root: Path, users: int = 500, days: int = 60,
                 n_boot: int = CONDITIONAL_BOOTSTRAP_SAMPLES, retailrocket: bool = True) -> Dict[str, Any]:
    """Runs every seed, aggregates, and optionally adds the RetailRocket section."""
    per_seed = {str(s): run_one_seed(s, Path(out_root) / f"seed_{s}", users, days, n_boot) for s in seeds}
    results: Dict[str, Any] = {
        "caveat": CAVEAT, "seeds": [int(s) for s in seeds], "users": users, "days": days,
        "blocks": {k: list(v) for k, v in FEATURE_BLOCKS.items()},
        "models": {k: v for k, v in model_feature_sets().items()},
        "aggregate": aggregate_seeds(per_seed), "per_seed": per_seed,
    }
    if retailrocket:
        results["retailrocket"] = run_retailrocket(Path(out_root), n_boot)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature-block ablation across simulation seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--n-boot", type=int, default=CONDITIONAL_BOOTSTRAP_SAMPLES)
    parser.add_argument("--no-retailrocket", action="store_true")
    args = parser.parse_args()

    from src.release.reporting import write_report

    res = run_ablation(args.seeds, Path(args.out), args.users, args.days, args.n_boot,
                       retailrocket=not args.no_retailrocket)
    print(f"Wrote {write_report(REPORT_NAME, RELEASE, res)}")

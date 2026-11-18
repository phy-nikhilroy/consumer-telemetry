"""RetailRocket disjoint-split re-run and the one-tree question (E10).

Builds train and test matrices from the RetailRocket warehouse (read only) on a split whose
target windows do not overlap, then fits LightGBM over a grid of scale_pos_weight, early-stopping
metric and patience, recording best_iteration, test ROC-AUC, PR-AUC and the single-feature
baseline per cell. Every cell writes only under its own directory in data/experiments/v1_1_rr.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import PREDICTION_WINDOW_DAYS, PRIMARY_TARGET, PROJECT_ROOT
from src.features.feature_store import (
    _parse_ref,
    generate_train_test_datasets,
    get_modeling_groups,
    get_modeling_matrices,
    validate_split,
)
from src.models.baselines import DEFAULT_SINGLE_FEATURE, single_feature_scores
from src.models.propensity import DEFAULT_PARAMS, compute_scale_pos_weight, grouped_validation_split

RR_DB_PATH = PROJECT_ROOT / "data" / "external" / "retailrocket.duckdb"
RR_OUT_DIR = PROJECT_ROOT / "data" / "experiments" / "v1_1_rr"
RR_TRAIN_REFS: Tuple[str, ...] = ("2015-08-24 00:00:00", "2015-08-31 00:00:00")
RR_TEST_REF = "2015-09-07 00:00:00"
RR_METRICS: Tuple[str, ...] = ("binary_logloss", "average_precision")
RR_PATIENCES: Tuple[int, ...] = (50, 200)
RR_REPORT_NAME = "v1_1_retailrocket"
RR_RELEASE = "1.1"


def resolve_dates(
    train_refs: Sequence[str], test_ref: str, max_event, window_days: int = PREDICTION_WINDOW_DAYS
) -> Tuple[List[str], str, int]:
    """Shifts all dates earlier by whole weeks until validate_split passes; returns (train, test, weeks)."""
    for weeks in range(0, 60):
        shift = timedelta(weeks=weeks)
        trains = [(_parse_ref(r) - shift).strftime("%Y-%m-%d %H:%M:%S") for r in train_refs]
        test = (_parse_ref(test_ref) - shift).strftime("%Y-%m-%d %H:%M:%S")
        try:
            validate_split(trains, test, prediction_window_days=window_days, max_event_timestamp=max_event)
            return trains, test, weeks
        except ValueError:
            continue
    raise ValueError("no valid split found within 60 weeks of shifting")


def cell_name(spw_mode: str, metric: str, patience: int) -> str:
    """Directory-safe cell identifier."""
    return f"spw-{spw_mode}__{metric}__pat{patience}"


def grid_cells() -> List[Dict[str, Any]]:
    """The 8 grid cells: spw mode x early-stopping metric x patience."""
    return [
        {"name": cell_name(m, e, p), "spw_mode": m, "metric": e, "patience": p}
        for m, e, p in itertools.product(("ratio", "one"), RR_METRICS, RR_PATIENCES)
    ]


def fit_cell(
    X_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray, spw_mode: str, metric: str, patience: int
) -> lgb.LGBMClassifier:
    """Fits one grid cell with the propensity parameters and a grouped early-stopping slice."""
    params = dict(DEFAULT_PARAMS)
    params["metric"] = metric
    params["scale_pos_weight"] = compute_scale_pos_weight(y_train) if spw_mode == "ratio" else 1.0
    fit_idx, val_idx = grouped_validation_split(y_train, groups)
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train[fit_idx], y_train[fit_idx],
        eval_X=X_train[val_idx], eval_y=y_train[val_idx],
        eval_metric=metric,
        callbacks=[lgb.early_stopping(stopping_rounds=patience, first_metric_only=True, verbose=False),
                   lgb.log_evaluation(period=0)],
    )
    return model


def run_cell(cell: Dict[str, Any], features_dir: Path, artifacts_dir: Path) -> Dict[str, Any]:
    """Runs one cell from a feature directory and writes its metrics under `artifacts_dir`."""
    X_tr, y_tr, names = get_modeling_matrices(split="train", target_col=PRIMARY_TARGET, features_dir=features_dir)
    X_te, y_te, _ = get_modeling_matrices(split="test", target_col=PRIMARY_TARGET, features_dir=features_dir)
    groups = get_modeling_groups("train", features_dir=features_dir)
    model = fit_cell(X_tr, y_tr, groups, cell["spw_mode"], cell["metric"], cell["patience"])
    proba = model.predict_proba(X_te)[:, 1]
    single = single_feature_scores(X_te, names)
    out = {
        **cell,
        "scale_pos_weight": float(model.get_params()["scale_pos_weight"]),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "pr_auc": float(average_precision_score(y_te, proba)),
        "single_feature": DEFAULT_SINGLE_FEATURE,
        "single_feature_roc_auc": float(roc_auc_score(y_te, single)),
        "single_feature_pr_auc": float(average_precision_score(y_te, single)),
        "train_positive_rate": float(np.mean(y_tr)),
        "test_positive_rate": float(np.mean(y_te)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    return out


def ensure_current_schema(db_path: Path) -> str:
    """
    Migrates the RetailRocket warehouse in place to the current schema and returns its
    version. The migration is idempotent; it adds the v2 columns and tables the feature SQL
    reads (`price_inr`, `discount_pct`, `fact_price_observations`) without touching rows.
    """
    from src.features.migrations import migrate

    con = duckdb.connect(str(db_path))
    try:
        return migrate(con)
    finally:
        con.close()


def run_grid(
    db_path: Path = RR_DB_PATH,
    out_dir: Path = RR_OUT_DIR,
    train_refs: Sequence[str] = RR_TRAIN_REFS,
    test_ref: str = RR_TEST_REF,
) -> Dict[str, Any]:
    """
    Upgrades the warehouse to the current schema, verifies the data range, validates the
    split, builds features once, and runs all cells.
    """
    ensure_current_schema(db_path)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        lo, hi = con.execute("SELECT MIN(timestamp), MAX(timestamp) FROM fact_events").fetchone()
    finally:
        con.close()
    trains, test, shift = resolve_dates(train_refs, test_ref, hi)
    out_dir = Path(out_dir)
    cells = grid_cells()
    first = out_dir / cells[0]["name"] / "features"
    generate_train_test_datasets(trains, test, db_path=db_path, features_dir=first)
    results = []
    for cell in cells:
        cdir = out_dir / cell["name"]
        fdir = cdir / "features"
        if fdir != first:
            shutil.copytree(first, fdir, dirs_exist_ok=True)
        results.append(run_cell(cell, fdir, cdir / "artifacts"))
    return {
        "data_range": [str(lo), str(hi)],
        "dates": {"train_refs": trains, "test_ref": test, "weeks_shifted": shift},
        "grid": results,
        "conclusion": "",
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: runs the grid and writes reports/v1_1_retailrocket.json."""
    from src.release.reporting import write_report

    parser = argparse.ArgumentParser(description="RetailRocket disjoint re-run and one-tree grid.")
    parser.add_argument("--db-path", type=Path, default=RR_DB_PATH)
    parser.add_argument("--out-dir", type=Path, default=RR_OUT_DIR)
    args = parser.parse_args(argv)
    results = run_grid(args.db_path, args.out_dir)
    print(write_report(RR_REPORT_NAME, RR_RELEASE, results))


if __name__ == "__main__":
    main()

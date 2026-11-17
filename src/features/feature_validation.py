"""
Feature Store EDA, Quality Validation, and Feature Selection for Downstream Modeling.
Performs data quality audits, correlation analysis, and tree-based importance ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure consumer-telemetry root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier

from src.config import TRAIN_FEATURES_PATH, SELECTED_FEATURES_PATH, rebase_path
from src.features.feature_store import fill_missing_features

DEFAULT_TRAIN_PATH = TRAIN_FEATURES_PATH
DEFAULT_SELECTED_PATH = SELECTED_FEATURES_PATH


def audit_feature_quality(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Performs audit on feature matrix:
    - Null/missing value percentages per feature.
    - Infinite value presence.
    - Constant feature (zero variance) detection.
    """
    total_rows = df.height
    audit_results: Dict[str, Any] = {
        "total_rows": total_rows,
        "total_columns": df.width,
        "missing_summary": {},
        "infinite_columns": [],
        "zero_variance_columns": [],
        "is_clean": True,
    }

    numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]

    for col in df.columns:
        null_count = df[col].null_count()
        null_pct = round((null_count / total_rows) * 100.0, 2)
        if null_count > 0:
            audit_results["missing_summary"][col] = {
                "null_count": null_count,
                "null_pct": null_pct,
            }

    for col in numeric_cols:
        series = df[col].drop_nulls()
        if len(series) > 0:
            arr = series.to_numpy()
            if np.isinf(arr).any():
                audit_results["infinite_columns"].append(col)
                audit_results["is_clean"] = False
            if np.nanstd(arr) == 0.0:
                audit_results["zero_variance_columns"].append(col)

    if audit_results["infinite_columns"]:
        audit_results["is_clean"] = False

    return audit_results


def analyze_correlations_and_redundancy(
    df: pl.DataFrame,
    target_col: str = "will_purchase_next_7d",
    collinearity_threshold: float = 0.90
) -> Tuple[Dict[str, float], List[Tuple[str, str, float]]]:
    """
    Computes Pearson correlation against target variable and detects multi-collinear pairs.
    """
    feature_cols = [
        c for c in df.columns
        if c not in ["user_id", "reference_date", "will_purchase_next_7d", "num_purchases_next_7d", "will_switch_app_next_7d", "top_category_l1"]
    ]

    # Nulls are filled exactly as the modeling matrices fill them, so selection judges
    # the same encoding the estimators will see.
    pdf = fill_missing_features(df, feature_cols + [target_col]).to_pandas()

    # Correlation with target (fill NaN from zero-variance columns with 0.0)
    corr_matrix = pdf.corr().fillna(0.0)
    target_corr = corr_matrix[target_col].drop(index=[target_col]).to_dict()
    sorted_target_corr = {
        k: (0.0 if np.isnan(v) else float(v))
        for k, v in target_corr.items()
    }
    sorted_target_corr = dict(sorted(sorted_target_corr.items(), key=lambda item: abs(item[1]), reverse=True))

    # Identify multi-collinear pairs (|r| >= threshold)
    collinear_pairs: List[Tuple[str, str, float]] = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            f1, f2 = feature_cols[i], feature_cols[j]
            r = corr_matrix.loc[f1, f2]
            if abs(r) >= collinearity_threshold:
                collinear_pairs.append((f1, f2, round(float(r), 4)))

    return sorted_target_corr, collinear_pairs


def compute_feature_importances(
    df: pl.DataFrame,
    target_col: str = "will_purchase_next_7d",
    random_state: int = 42
) -> List[Tuple[str, float]]:
    """
    Trains a baseline tree-based ensemble to extract Gini importance scores.
    """
    feature_cols = [
        c for c in df.columns
        if c not in ["user_id", "reference_date", "will_purchase_next_7d", "num_purchases_next_7d", "will_switch_app_next_7d", "top_category_l1"]
    ]

    X = fill_missing_features(df, feature_cols).to_pandas()
    y = df[target_col].to_numpy()

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X, y)

    scores = [
        (col, round(float(imp), 4))
        for col, imp in zip(feature_cols, model.feature_importances_)
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def select_features(
    df: pl.DataFrame,
    importance_scores: List[Tuple[str, float]],
    collinear_pairs: List[Tuple[str, str, float]],
    top_k: int = 18,
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Filters features by importance while pruning highly redundant collinear pairs.
    Saves selected feature schema to JSON.
    """
    # Track pairs to prune (drop lower importance feature in collinear pair)
    importance_dict = dict(importance_scores)
    prune_set = set()

    for f1, f2, r in collinear_pairs:
        imp1 = importance_dict.get(f1, 0.0)
        imp2 = importance_dict.get(f2, 0.0)
        if imp1 >= imp2:
            prune_set.add(f2)
        else:
            prune_set.add(f1)

    selected = []
    for feat, score in importance_scores:
        if feat not in prune_set and len(selected) < top_k:
            selected.append(feat)

    # If top_k not reached due to pruning, fill from remaining
    for feat, score in importance_scores:
        if feat not in selected and len(selected) < top_k:
            selected.append(feat)

    result = {
        "total_candidate_features": len(importance_scores),
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "pruned_collinear_features": list(prune_set),
        "top_predictive_features": [
            {"feature": f, "importance": float(importance_dict[f])}
            for f in selected[:5]
        ],
        "target_variables": [
            "will_purchase_next_7d",
            "num_purchases_next_7d",
            "will_switch_app_next_7d"
        ],
    }

    out_file = Path(output_path) if output_path else DEFAULT_SELECTED_PATH
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def run_feature_validation_pipeline(
    train_path: Optional[Union[str, Path]] = None,
    output_json_path: Optional[Union[str, Path]] = None,
    features_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Executes feature audit, correlation analysis, tree importance, and selection.

    `features_dir` points both the input matrix and the output selection at another
    dataset's feature directory; explicit paths win over it.
    """
    in_path = Path(train_path) if train_path else rebase_path(DEFAULT_TRAIN_PATH, features_dir)
    out_path = (
        Path(output_json_path) if output_json_path
        else rebase_path(DEFAULT_SELECTED_PATH, features_dir)
    )
    print(f"Loading feature matrix from {in_path}...")
    df = pl.read_parquet(in_path)

    print("\n--- 1. Data Quality and Audit ---")
    audit = audit_feature_quality(df)
    print(f"Total Rows: {audit['total_rows']}, Total Columns: {audit['total_columns']}")
    print(f"Missing Values: {len(audit['missing_summary'])} columns with nulls")
    print(f"Infinite Values: {len(audit['infinite_columns'])}")
    print(f"Matrix Status: {'CLEAN (Valid)' if audit['is_clean'] else 'DIRTY (Has Errors)'}")

    print("\n--- 2. Correlation and Multi-Collinearity Analysis ---")
    target_corrs, collinear = analyze_correlations_and_redundancy(df)
    print(f"Detected {len(collinear)} multi-collinear feature pairs (|r| >= 0.90):")
    for f1, f2, r in collinear[:5]:
        print(f"  {f1} <-> {f2}: r = {r}")

    print("\n--- 3. Baseline Tree-Based Feature Importance ---")
    importances = compute_feature_importances(df)
    print("Top 8 Features by Tree Importance:")
    for feat, imp in importances[:8]:
        print(f"  {feat:<32}: {imp:.4f}")

    print("\n--- 4. Feature Selection ---")
    selection = select_features(
        df=df,
        importance_scores=importances,
        collinear_pairs=collinear,
        output_path=out_path
    )
    print(f"Selected {selection['selected_feature_count']} features. Saved to {out_path}")

    print("\n--- 5. Top 3 Most Predictive Features Summary ---")
    for i, item in enumerate(selection["top_predictive_features"][:3], 1):
        feat = item["feature"]
        imp = item["importance"]
        print(f"{i}. {feat} (importance: {imp:.4f})")

    return {
        "audit": audit,
        "selected_features": selection,
        "top_features": selection["top_predictive_features"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Store EDA and Selection Pipeline.")
    parser.add_argument("--train-path", type=str, default=None, help="Train feature parquet path")
    parser.add_argument("--output-json", type=str, default=None, help="Output selected features JSON path")
    parser.add_argument("--features-dir", type=str, default=None,
                        help="Read and write this feature directory instead of data/features")
    args = parser.parse_args()

    run_feature_validation_pipeline(
        train_path=args.train_path,
        output_json_path=args.output_json,
        features_dir=args.features_dir,
    )

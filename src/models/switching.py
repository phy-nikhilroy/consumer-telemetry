"""
App switching model with SHAP attribution.

Two complementary formulations of the same business question are trained here.

The binary model predicts `will_switch_app_next_7d`: will this user's next purchase
land on an app other than their historical dominant app?

The three-class model matches the richer framing of the specification. For a user who
has been browsing, the outcome is one of:

  class 0  no_purchase             the user does not buy at all
  class 1  bought_on_dominant_app  the user buys on the app they already favour
  class 2  switched_app            the user buys, but on a different app

The three-class labels are derived from the existing forward targets, so they inherit
the same leak-free reference-date cutoff as every other feature store column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure consumer-telemetry root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS_DIR,
    SWITCHING_CONDITIONAL_TARGET,
    PRIMARY_TARGET,
    RANDOM_SEED,
    CONDITIONAL_BOOTSTRAP_SAMPLES,
    SELECTED_FEATURES_PATH,
    SWITCHING_CONDITIONAL_MODEL_PATH,
    SWITCHING_MODEL_PATH,
    SWITCHING_TARGET,
    TEST_FEATURES_PATH,
    TRAIN_FEATURES_PATH,
    rebase_path,
)
from src.features.feature_store import fill_missing_features, get_modeling_groups
from src.models.propensity import grouped_validation_split

MULTICLASS_LABELS: Tuple[str, ...] = ("no_purchase", "bought_on_dominant_app", "switched_app")

SWITCHING_MULTICLASS_PATH = ARTIFACTS_DIR / "switching_lgbm_multiclass.joblib"
SHAP_SUMMARY_PLOT_PATH = ARTIFACTS_DIR / "switching_shap_summary.png"
SHAP_BAR_PLOT_PATH = ARTIFACTS_DIR / "switching_shap_importance.png"
SWITCHING_CONFUSION_PLOT_PATH = ARTIFACTS_DIR / "switching_confusion_matrix.png"
SWITCHING_METRICS_PATH = ARTIFACTS_DIR / "switching_metrics.json"

BINARY_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

MULTICLASS_PARAMS: Dict[str, Any] = {
    **BINARY_PARAMS,
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": len(MULTICLASS_LABELS),
}


def load_selected_feature_names(
    df: pl.DataFrame,
    selected_features_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Returns the pruned, non-collinear feature list intersected with the given matrix."""
    path = Path(selected_features_path) if selected_features_path else SELECTED_FEATURES_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
        names = info["selected_features"]
        if info.get("schema_version") == 2:
            names = info.get("features_by_target", {}).get(SWITCHING_TARGET, names)
        return [name for name in names if name in df.columns]

    ignore = {
        "user_id",
        "reference_date",
        "top_category_l1",
        "will_purchase_next_7d",
        "num_purchases_next_7d",
        "will_switch_app_next_7d",
    }
    return [c for c in df.columns if c not in ignore]


def build_multiclass_target(df: pl.DataFrame) -> np.ndarray:
    """
    Derives the three-class purchase destination label from the forward targets.

    A switch implies a purchase, so the switch flag is checked first and the dominant-app
    class is everything else that purchased.
    """
    purchased = df[PRIMARY_TARGET].to_numpy()
    switched = df[SWITCHING_TARGET].to_numpy()

    labels = np.zeros(df.height, dtype=int)
    labels[(purchased == 1) & (switched == 0)] = 1
    labels[switched == 1] = 2
    return labels


def get_switching_matrices(
    split: str = "train",
    features_path: Optional[Union[str, Path]] = None,
    features_dir: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Loads one split and returns (X, y_binary, y_multiclass, feature_names).

    `features_dir` reads the split and its feature selection from another dataset's
    feature directory; an explicit `features_path` wins over it.
    """
    if features_path is None:
        default = TRAIN_FEATURES_PATH if split == "train" else TEST_FEATURES_PATH
        features_path = rebase_path(default, features_dir)

    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix not found at {path}")

    df = pl.read_parquet(path)
    feature_names = load_selected_feature_names(
        df, rebase_path(SELECTED_FEATURES_PATH, features_dir)
    )

    X = fill_missing_features(df, feature_names).to_numpy().astype(np.float32)
    y_binary = df[SWITCHING_TARGET].to_numpy()
    y_multiclass = build_multiclass_target(df)

    return X, y_binary, y_multiclass, feature_names


def train_switching_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    validation_fraction: float = 0.2,
    early_stopping_rounds: int = 50,
    random_state: int = RANDOM_SEED,
    groups: Optional[np.ndarray] = None,
) -> lgb.LGBMClassifier:
    """
    Fits a LightGBM classifier for either the binary or the three-class formulation.

    The objective is inferred from the label cardinality, and a stratified slice of the
    training matrix selects the boosting round count. When `groups` is given the slice is
    a grouped split so no user spans fit and validation.
    """
    n_classes = len(np.unique(y_train))
    base = dict(MULTICLASS_PARAMS if n_classes > 2 else BINARY_PARAMS)
    if n_classes > 2:
        base["num_class"] = n_classes
    if params:
        base.update(params)

    eval_metric = "multi_logloss" if n_classes > 2 else "binary_logloss"

    if groups is not None:
        fit_idx, val_idx = grouped_validation_split(y_train, groups, random_state=random_state)
        X_fit, X_val = X_train[fit_idx], X_train[val_idx]
        y_fit, y_val = y_train[fit_idx], y_train[val_idx]
    else:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=validation_fraction,
            stratify=y_train,
            random_state=random_state,
        )

    model = lgb.LGBMClassifier(**base)
    model.fit(
        X_fit,
        y_fit,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric=eval_metric,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds, first_metric_only=True, verbose=False
            ),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def evaluate_binary_switching(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Scores the binary switching model."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "n_samples": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": classification_report(
            y, pred, target_names=["stayed", "switched"], zero_division=0
        ),
    }


def select_threshold_by_f1(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    y: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> float:
    """
    Picks the decision threshold that maximizes F1, measured on the data passed in.

    This must be called with the training split. The default 0.5 cutoff is only optimal
    when the classes are balanced and the model is calibrated; for a minority positive
    class it systematically under-predicts the positive label and suppresses recall.
    Tuning the threshold on train and then applying it unchanged to test keeps the
    reported test metrics honest.
    """
    candidates = grid if grid is not None else np.linspace(0.05, 0.95, 91)
    proba = model.predict_proba(X)[:, 1]

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        score = f1_score(y, (proba >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold


def evaluate_multiclass_switching(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """Scores the three-class purchase destination model."""
    proba = model.predict_proba(X)
    pred = np.argmax(proba, axis=1)
    present = sorted(np.unique(np.concatenate([y, pred])).tolist())

    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "multi_log_loss": float(log_loss(y, proba, labels=list(range(proba.shape[1])))),
        "class_distribution": {
            MULTICLASS_LABELS[c]: int(np.sum(y == c)) for c in range(len(MULTICLASS_LABELS))
        },
        "confusion_matrix": confusion_matrix(
            y, pred, labels=list(range(len(MULTICLASS_LABELS)))
        ).tolist(),
        "labels": list(MULTICLASS_LABELS),
        "classification_report": classification_report(
            y,
            pred,
            labels=present,
            target_names=[MULTICLASS_LABELS[c] for c in present],
            zero_division=0,
        ),
    }


def plot_confusion_matrix(
    matrix: List[List[int]],
    labels: Tuple[str, ...] = MULTICLASS_LABELS,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Saves an annotated confusion matrix heatmap for the three-class model."""
    out = Path(output_path) if output_path else SWITCHING_CONFUSION_PLOT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(arr, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Purchase Destination: Confusion Matrix (Test Split)")

    threshold = arr.max() / 2.0 if arr.max() > 0 else 0.5
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j,
                i,
                int(arr[i, j]),
                ha="center",
                va="center",
                color="white" if arr[i, j] > threshold else "black",
            )

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _normalize_shap_values(raw: Any, class_index: int = 1) -> np.ndarray:
    """
    Reduces SHAP output to a single 2-D (samples, features) array.

    Depending on version and model type, TreeExplainer returns either a list of arrays
    per class, a 3-D array with a trailing class axis, or a plain 2-D array for a single
    output. All three shapes are handled here so the plotting code stays simple.
    """
    if isinstance(raw, list):
        idx = class_index if len(raw) > class_index else -1
        return np.asarray(raw[idx])

    arr = np.asarray(raw)
    if arr.ndim == 3:
        idx = class_index if arr.shape[2] > class_index else -1
        return arr[:, :, idx]
    return arr


def compute_shap_values(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    class_index: int = 1,
) -> np.ndarray:
    """Returns the SHAP value matrix for the requested class."""
    explainer = shap.TreeExplainer(model)
    return _normalize_shap_values(explainer.shap_values(X), class_index=class_index)


def rank_shap_drivers(
    shap_values: np.ndarray,
    feature_names: List[str],
    X: np.ndarray,
    top_n: int = 5,
) -> pl.DataFrame:
    """
    Ranks features by mean absolute SHAP value and reports their directional effect.

    The direction column is the correlation between a feature's value and its SHAP
    contribution: positive means higher values of the feature push the prediction toward
    switching, negative means they push against it.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    rows: List[Dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        feature_col = X[:, idx].astype(float)
        shap_col = shap_values[:, idx].astype(float)
        if np.std(feature_col) > 0 and np.std(shap_col) > 0:
            direction = float(np.corrcoef(feature_col, shap_col)[0, 1])
        else:
            direction = 0.0
        rows.append(
            {
                "rank": rank,
                "feature": feature_names[idx],
                "mean_abs_shap": float(mean_abs[idx]),
                "direction": direction,
                "effect": "increases switching" if direction > 0 else "reduces switching",
            }
        )
    return pl.DataFrame(rows)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    max_display: int = 15,
) -> Path:
    """Saves the SHAP beeswarm summary plot."""
    out = Path(output_path) if output_path else SHAP_SUMMARY_PLOT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("App Switching: SHAP Value Distribution", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    return out


def plot_shap_importance_bar(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    max_display: int = 15,
) -> Path:
    """Saves the mean absolute SHAP importance bar chart."""
    out = Path(output_path) if output_path else SHAP_BAR_PLOT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title("App Switching: Mean Absolute SHAP Importance", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    return out


def build_conditional_target(df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns the rows eligible for the conditional switching model with the derived target.

    `switched_given_purchase` is defined only where `will_purchase_next_7d == 1`; its value
    is `will_switch_app_next_7d`. Non-buyer rows are dropped.
    """
    return df.filter(pl.col(PRIMARY_TARGET) == 1).with_columns(
        pl.col(SWITCHING_TARGET).alias(SWITCHING_CONDITIONAL_TARGET)
    )


def bootstrap_auc_ci(
    y: np.ndarray,
    score: np.ndarray,
    n_boot: int = CONDITIONAL_BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
    random_state: int = RANDOM_SEED,
) -> List[float]:
    """Percentile bootstrap confidence interval for ROC-AUC; [nan, nan] if undefined."""
    y = np.asarray(y)
    score = np.asarray(score)
    rng = np.random.default_rng(random_state)
    stats: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(float(roc_auc_score(y[idx], score[idx])))
    if not stats:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def evaluate_conditional_switching(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    dominant_share: np.ndarray,
) -> Dict[str, Any]:
    """
    Scores the conditional model on buyer rows against the `1 - dominant_app_share` baseline.

    `delta_vs_baseline` is model ROC-AUC minus baseline ROC-AUC.
    """
    proba = model.predict_proba(X)[:, 1]
    baseline = 1.0 - np.asarray(dominant_share, dtype=float)
    roc = float(roc_auc_score(y, proba))
    base_auc = float(roc_auc_score(y, baseline))
    return {
        "n_samples": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "roc_auc": roc,
        "pr_auc": float(average_precision_score(y, proba)),
        "ci": bootstrap_auc_ci(y, proba),
        "baseline_dominant_share_auc": base_auc,
        "delta_vs_baseline": roc - base_auc,
    }


def run_conditional_switching(
    features_dir: Optional[Union[str, Path]] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Trains the buyers-only switching model, saves it, and returns test metrics and the model."""
    train_df = pl.read_parquet(rebase_path(TRAIN_FEATURES_PATH, features_dir))
    test_df = pl.read_parquet(rebase_path(TEST_FEATURES_PATH, features_dir))
    names = load_selected_feature_names(train_df, rebase_path(SELECTED_FEATURES_PATH, features_dir))

    tr = build_conditional_target(train_df)
    te = build_conditional_target(test_df)
    X_tr = fill_missing_features(tr, names).to_numpy().astype(np.float32)
    X_te = fill_missing_features(te, names).to_numpy().astype(np.float32)
    y_tr = tr[SWITCHING_CONDITIONAL_TARGET].to_numpy()
    y_te = te[SWITCHING_CONDITIONAL_TARGET].to_numpy()

    model = train_switching_model(X_tr, y_tr, groups=tr["user_id"].to_numpy())
    metrics = evaluate_conditional_switching(
        model, X_te, y_te, te["dominant_app_share"].to_numpy()
    )

    out = Path(model_path) if model_path else rebase_path(
        SWITCHING_CONDITIONAL_MODEL_PATH, artifacts_dir
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"\n--- Conditional Switching (buyers only) ---\n  {metrics}\n  Saved to {out}")
    return {"model": model, "metrics": metrics, "feature_names": names, "path": out}


def run_switching_pipeline(
    model_path: Optional[Union[str, Path]] = None,
    multiclass_path: Optional[Union[str, Path]] = None,
    metrics_path: Optional[Union[str, Path]] = None,
    top_n_drivers: int = 5,
    features_dir: Optional[Union[str, Path]] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Trains, evaluates, explains, and persists both switching formulations.

    `features_dir` and `artifacts_dir` redirect inputs and outputs to another dataset's
    directories; explicit paths win over them.
    """
    X_train, y_bin_train, y_mc_train, feature_names = get_switching_matrices(
        "train", features_dir=features_dir
    )
    X_test, y_bin_test, y_mc_test, test_names = get_switching_matrices(
        "test", features_dir=features_dir
    )

    if feature_names != test_names:
        raise ValueError("Train and test feature ordering differ; refusing to train.")

    print("--- Binary App Switching Model ---")
    print(f"  Train: {X_train.shape[0]} rows x {X_train.shape[1]} features, "
          f"switch rate {float(np.mean(y_bin_train)):.3f}")
    print(f"  Test : {X_test.shape[0]} rows x {X_test.shape[1]} features, "
          f"switch rate {float(np.mean(y_bin_test)):.3f}")

    groups = get_modeling_groups("train", features_dir=features_dir)
    binary_model = train_switching_model(X_train, y_bin_train, groups=groups)
    binary_train_metrics = evaluate_binary_switching(binary_model, X_train, y_bin_train)
    binary_test_metrics = evaluate_binary_switching(binary_model, X_test, y_bin_test)

    # Switching is a minority outcome, so the default 0.5 cutoff buys precision at a
    # heavy cost in recall. The threshold is tuned on train only, then applied to test.
    tuned_threshold = select_threshold_by_f1(binary_model, X_train, y_bin_train)
    binary_test_tuned = evaluate_binary_switching(
        binary_model, X_test, y_bin_test, threshold=tuned_threshold
    )

    for label, metrics in (
        ("TRAIN  (t=0.50)", binary_train_metrics),
        ("TEST   (t=0.50)", binary_test_metrics),
        (f"TEST   (t={tuned_threshold:.2f})", binary_test_tuned),
    ):
        cm = metrics["confusion_matrix"]
        print(f"\n  {label}: ROC-AUC {metrics['roc_auc']:.4f} | PR-AUC {metrics['pr_auc']:.4f} | "
              f"F1 {metrics['f1']:.4f} | Accuracy {metrics['accuracy']:.4f} | "
              f"Log Loss {metrics['log_loss']:.4f}")
        print(f"         TN={cm['true_negative']} FP={cm['false_positive']} "
              f"FN={cm['false_negative']} TP={cm['true_positive']}")

    print(f"\n  Threshold tuned on train: {tuned_threshold:.2f} "
          f"(F1 {binary_test_metrics['f1']:.4f} -> {binary_test_tuned['f1']:.4f} on test)")
    print("\n" + binary_test_tuned["classification_report"])

    print("\n--- Three-Class Purchase Destination Model ---")
    distribution = {MULTICLASS_LABELS[c]: int(np.sum(y_mc_train == c)) for c in range(3)}
    print(f"  Train class distribution: {distribution}")

    multiclass_model = train_switching_model(X_train, y_mc_train, groups=groups)
    mc_test_metrics = evaluate_multiclass_switching(multiclass_model, X_test, y_mc_test)

    print(f"  TEST: Accuracy {mc_test_metrics['accuracy']:.4f} | "
          f"Macro F1 {mc_test_metrics['macro_f1']:.4f} | "
          f"Multi Log Loss {mc_test_metrics['multi_log_loss']:.4f}")
    print("\n" + mc_test_metrics["classification_report"])

    conditional = run_conditional_switching(
        features_dir=features_dir, artifacts_dir=artifacts_dir
    )

    print("--- SHAP Attribution (Binary Switching Model, Test Split) ---")
    shap_values = compute_shap_values(binary_model, X_test, class_index=1)
    drivers = rank_shap_drivers(shap_values, feature_names, X_test, top_n=top_n_drivers)
    print(drivers)

    print("\n  Top switching drivers:")
    for row in drivers.head(3).iter_rows(named=True):
        print(f"    {row['rank']}. {row['feature']} "
              f"(mean |SHAP| {row['mean_abs_shap']:.4f}, {row['effect']})")

    b_out = Path(model_path) if model_path else rebase_path(SWITCHING_MODEL_PATH, artifacts_dir)
    m_out = (
        Path(multiclass_path) if multiclass_path
        else rebase_path(SWITCHING_MULTICLASS_PATH, artifacts_dir)
    )
    b_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(binary_model, b_out)
    joblib.dump(multiclass_model, m_out)

    summary_plot = plot_shap_summary(
        shap_values, X_test, feature_names,
        output_path=rebase_path(SHAP_SUMMARY_PLOT_PATH, artifacts_dir),
    )
    bar_plot = plot_shap_importance_bar(
        shap_values, X_test, feature_names,
        output_path=rebase_path(SHAP_BAR_PLOT_PATH, artifacts_dir),
    )
    confusion_plot = plot_confusion_matrix(
        mc_test_metrics["confusion_matrix"],
        output_path=rebase_path(SWITCHING_CONFUSION_PLOT_PATH, artifacts_dir),
    )

    metrics_out = Path(metrics_path) if metrics_path else rebase_path(SWITCHING_METRICS_PATH, artifacts_dir)
    payload = {
        "binary_target": SWITCHING_TARGET,
        "feature_names": feature_names,
        "binary_train": {k: v for k, v in binary_train_metrics.items() if k != "classification_report"},
        "binary_test": {k: v for k, v in binary_test_metrics.items() if k != "classification_report"},
        "binary_test_tuned_threshold": {
            k: v for k, v in binary_test_tuned.items() if k != "classification_report"
        },
        "tuned_threshold": tuned_threshold,
        "multiclass_test": {k: v for k, v in mc_test_metrics.items() if k != "classification_report"},
        "conditional_test": conditional["metrics"],
        "top_shap_drivers": drivers.to_dicts(),
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved binary model to      {b_out}")
    print(f"Saved multiclass model to  {m_out}")
    print(f"Saved SHAP summary to      {summary_plot}")
    print(f"Saved SHAP bar chart to    {bar_plot}")
    print(f"Saved confusion matrix to  {confusion_plot}")
    print(f"Saved metrics to           {metrics_out}")

    return {
        "binary_model": binary_model,
        "multiclass_model": multiclass_model,
        "feature_names": feature_names,
        "binary_test_metrics": binary_test_metrics,
        "binary_test_metrics_tuned": binary_test_tuned,
        "tuned_threshold": tuned_threshold,
        "multiclass_test_metrics": mc_test_metrics,
        "shap_values": shap_values,
        "drivers": drivers,
        "X_test": X_test,
        "paths": {
            "binary_model": b_out,
            "multiclass_model": m_out,
            "shap_summary": summary_plot,
            "shap_bar": bar_plot,
            "confusion_matrix": confusion_plot,
            "metrics": metrics_out,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the app switching models with SHAP attribution.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of SHAP drivers to report")
    parser.add_argument("--features-dir", type=str, default=None, help="Feature directory to read")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Directory for outputs")
    args = parser.parse_args()

    run_switching_pipeline(
        top_n_drivers=args.top_n,
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
    )

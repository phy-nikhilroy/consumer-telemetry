"""
LightGBM purchase propensity model.

Trains a binary classifier that predicts whether a user will make at least one
purchase in the 7 days following the feature reference date. Features come from the
leak-free feature store; the test split uses a later reference date so its target
window does not overlap the training target window.
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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.config import (
    ARTIFACTS_DIR,
    PRIMARY_TARGET,
    PROPENSITY_MODEL_PATH,
    RANDOM_SEED,
    rebase_path,
)
from src.features.feature_store import get_modeling_groups, get_modeling_matrices
from src.models.baselines import baseline_report

# Conservative hyperparameters. The modeling matrix is a few hundred rows wide by
# roughly 18 features, so shallow trees and strong regularization matter more than
# capacity: an unconstrained LightGBM memorizes this matrix immediately.
DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    # Early stopping follows this metric. Using "auc" here caused the fit to halt after
    # four boosting rounds on an AUC plateau, leaving every predicted probability
    # compressed into [0.42, 0.60]. Optimizing log loss instead keeps the ranking
    # quality while producing probabilities that span the unit interval, which the
    # calibration and lift analysis downstream depends on.
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

FEATURE_IMPORTANCE_PLOT_PATH = ARTIFACTS_DIR / "propensity_feature_importance.png"
PROPENSITY_METRICS_PATH = ARTIFACTS_DIR / "propensity_metrics.json"


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """
    Returns the negative-to-positive ratio used to reweight the positive class.

    A value near 1.0 means the target is already close to balanced and the weight is
    effectively a no-op.
    """
    positives = float(np.sum(y == 1))
    negatives = float(np.sum(y == 0))
    if positives == 0.0:
        return 1.0
    return negatives / positives


def grouped_validation_split(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns `(fit_idx, val_idx)` from the first fold of a shuffled StratifiedGroupKFold.

    All rows of one group (a user) land on the same side, so the early-stopping slice
    never shares a user with the rows the model is fitted on.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fit_idx, val_idx = next(splitter.split(np.zeros(len(y)), y, groups))
    return fit_idx, val_idx


def train_propensity_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    validation_fraction: float = 0.2,
    early_stopping_rounds: int = 50,
    random_state: int = RANDOM_SEED,
    groups: Optional[np.ndarray] = None,
) -> lgb.LGBMClassifier:
    """
    Fits a LightGBM binary classifier with class reweighting and early stopping.

    When `groups` (user ids) is given, the validation slice is a grouped split so no user
    spans fit and validation; otherwise a stratified row split is used.

    A stratified slice of the training matrix is held back purely to choose the number
    of boosting rounds. The test split is never touched here, so the reported test
    metrics stay honest.
    """
    model_params = dict(DEFAULT_PARAMS)
    if params:
        model_params.update(params)
    model_params["scale_pos_weight"] = compute_scale_pos_weight(y_train)

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

    model = lgb.LGBMClassifier(**model_params)
    model.fit(
        X_fit,
        y_fit,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def evaluate_propensity_model(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Scores a fitted model and returns ranking, calibration, and threshold metrics.

    ROC-AUC and PR-AUC measure ranking quality and are threshold free. Log loss and
    the Brier score measure how well the predicted probabilities are calibrated.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    base_rate = float(np.mean(y))

    return {
        "n_samples": int(len(y)),
        "positive_rate": base_rate,
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "pr_auc": float(average_precision_score(y, y_proba)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y, y_proba, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, y_proba)),
        "threshold": threshold,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": classification_report(
            y, y_pred, target_names=["no_purchase", "purchase"], zero_division=0
        ),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
    }


def plot_feature_importance(
    model: lgb.LGBMClassifier,
    feature_names: List[str],
    top_n: int = 15,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Purchase Propensity: Top Feature Importances",
) -> Path:
    """Saves a horizontal bar chart of the top_n features by LightGBM gain."""
    out = Path(output_path) if output_path else FEATURE_IMPORTANCE_PLOT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    gains = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gains)[::-1][:top_n]
    labels = [feature_names[i] for i in order][::-1]
    values = [float(gains[i]) for i in order][::-1]

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.42 * len(labels))))
    ax.barh(labels, values, color="#2f6f9f")
    ax.set_xlabel("Total gain")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    return out


def save_model(model: Any, path: Optional[Union[str, Path]] = None) -> Path:
    """Serializes a fitted model to disk with joblib."""
    out = Path(path) if path else PROPENSITY_MODEL_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    return out


def load_model(path: Optional[Union[str, Path]] = None) -> Any:
    """Loads a serialized model from disk."""
    src = Path(path) if path else PROPENSITY_MODEL_PATH
    if not src.exists():
        raise FileNotFoundError(f"Model artifact not found at {src}")
    return joblib.load(src)


def print_evaluation(metrics: Dict[str, Any], split_label: str) -> None:
    """Prints a readable metrics block for one evaluation split."""
    cm = metrics["confusion_matrix"]
    print(f"\n--- {split_label} Metrics (n = {metrics['n_samples']}) ---")
    print(f"  Positive class rate : {metrics['positive_rate']:.3f}")
    print(f"  ROC-AUC             : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC              : {metrics['pr_auc']:.4f}")
    print(f"  F1 (threshold 0.50) : {metrics['f1']:.4f}")
    print(f"  Log Loss            : {metrics['log_loss']:.4f}")
    print(f"  Brier Score         : {metrics['brier_score']:.4f}")
    print(
        f"  Confusion matrix    : TN={cm['true_negative']} FP={cm['false_positive']} "
        f"FN={cm['false_negative']} TP={cm['true_positive']}"
    )
    print("\n" + metrics["classification_report"])


def run_training_pipeline(
    target_col: str = PRIMARY_TARGET,
    model_path: Optional[Union[str, Path]] = None,
    plot_path: Optional[Union[str, Path]] = None,
    metrics_path: Optional[Union[str, Path]] = None,
    top_n_features: int = 15,
    features_dir: Optional[Union[str, Path]] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Trains, evaluates, and persists the purchase propensity model end to end.

    `features_dir` and `artifacts_dir` redirect the inputs and outputs to another
    dataset's directories, so a second run cannot overwrite the committed artifacts.
    Explicit paths win over them.

    Returns a dictionary holding the fitted model, the train and test metric blocks,
    and the paths of every artifact written.
    """
    X_train, y_train, feature_names = get_modeling_matrices(
        split="train", target_col=target_col, features_dir=features_dir
    )
    X_test, y_test, test_feature_names = get_modeling_matrices(
        split="test", target_col=target_col, features_dir=features_dir
    )

    if feature_names != test_feature_names:
        raise ValueError("Train and test feature ordering differ; refusing to train.")

    print(f"Training purchase propensity model on target '{target_col}'")
    print(f"  Train matrix: {X_train.shape[0]} rows x {X_train.shape[1]} features")
    print(f"  Test matrix : {X_test.shape[0]} rows x {X_test.shape[1]} features")
    print(f"  Train positive rate: {float(np.mean(y_train)):.3f}")
    print(f"  Test positive rate : {float(np.mean(y_test)):.3f}")
    print(f"  scale_pos_weight   : {compute_scale_pos_weight(y_train):.4f}")

    groups = get_modeling_groups("train", features_dir=features_dir)
    model = train_propensity_model(X_train, y_train, groups=groups)

    train_metrics = evaluate_propensity_model(model, X_train, y_train)
    test_metrics = evaluate_propensity_model(model, X_test, y_test)

    print(f"\nBest boosting iteration: {test_metrics['best_iteration']}")
    print_evaluation(train_metrics, "TRAIN")
    print_evaluation(test_metrics, "TEST")

    saved_model = save_model(model, model_path or rebase_path(PROPENSITY_MODEL_PATH, artifacts_dir))
    saved_plot = plot_feature_importance(
        model,
        feature_names,
        top_n=top_n_features,
        output_path=plot_path or rebase_path(FEATURE_IMPORTANCE_PLOT_PATH, artifacts_dir),
    )

    m_out = Path(metrics_path) if metrics_path else rebase_path(PROPENSITY_METRICS_PATH, artifacts_dir)
    m_out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "target": target_col,
        "feature_names": feature_names,
        "train": {k: v for k, v in train_metrics.items() if k != "classification_report"},
        "test": {
            **{k: v for k, v in test_metrics.items() if k != "classification_report"},
            **baseline_report(
                y_test, model.predict_proba(X_test)[:, 1],
                X_train, y_train, X_test, feature_names,
            ),
        },
    }
    with open(m_out, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nSaved model to            {saved_model}")
    print(f"Saved importance plot to  {saved_plot}")
    print(f"Saved metrics to          {m_out}")

    return {
        "model": model,
        "feature_names": feature_names,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_path": saved_model,
        "plot_path": saved_plot,
        "metrics_path": m_out,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LightGBM purchase propensity model.")
    parser.add_argument("--target", type=str, default=PRIMARY_TARGET, help="Binary target column")
    parser.add_argument("--top-n", type=int, default=15, help="Features shown in the importance plot")
    parser.add_argument("--features-dir", type=str, default=None, help="Feature directory to read")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Directory for model outputs")
    args = parser.parse_args()

    run_training_pipeline(
        target_col=args.target,
        top_n_features=args.top_n,
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
    )

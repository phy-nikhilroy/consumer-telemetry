"""
Probability calibration, decile lift, and cumulative gains analysis.

A classifier can rank users well while still emitting probabilities that do not match
observed frequencies. Ranking quality is what lift and gains charts measure; calibration
is what makes a predicted score usable as an actual expected purchase rate in a business
rule. This module measures and corrects both, and writes the charts that the evaluation
notebooks render.
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
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import (
    ARTIFACTS_DIR,
    PRIMARY_TARGET,
    PROPENSITY_CALIBRATED_PATH,
    RANDOM_SEED,
)
from src.features.feature_store import get_modeling_matrices
from src.models.propensity import (
    DEFAULT_PARAMS,
    compute_scale_pos_weight,
    train_propensity_model,
)

CALIBRATION_PLOT_PATH = ARTIFACTS_DIR / "propensity_calibration_curve.png"
LIFT_PLOT_PATH = ARTIFACTS_DIR / "propensity_lift_chart.png"
GAINS_PLOT_PATH = ARTIFACTS_DIR / "propensity_cumulative_gains.png"
EVALUATION_SUMMARY_PATH = ARTIFACTS_DIR / "propensity_evaluation_summary.json"

CALIBRATION_METHODS: Tuple[str, ...] = ("isotonic", "sigmoid")


def build_base_estimator(y_train: np.ndarray, n_estimators: int = 50) -> lgb.LGBMClassifier:
    """
    Returns an unfitted LightGBM estimator configured like the propensity model.

    A fixed boosting round count replaces early stopping here, because
    CalibratedClassifierCV refits the estimator on internal cross-validation folds and
    cannot pass an evaluation set through to each fit.
    """
    params = dict(DEFAULT_PARAMS)
    params["n_estimators"] = n_estimators
    params["scale_pos_weight"] = compute_scale_pos_weight(y_train)
    return lgb.LGBMClassifier(**params)


def select_calibration_method(
    X_train: np.ndarray,
    y_train: np.ndarray,
    methods: Tuple[str, ...] = CALIBRATION_METHODS,
    cv_folds: int = 5,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Chooses a calibration method using cross-validated Brier score on the training data.

    Selecting on the training matrix rather than the test split keeps the reported test
    metrics free of selection bias.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores: Dict[str, Dict[str, float]] = {}

    base_proba = cross_val_predict(
        build_base_estimator(y_train), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    scores["uncalibrated"] = {
        "brier_score": float(brier_score_loss(y_train, base_proba)),
        "log_loss": float(log_loss(y_train, base_proba, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_train, base_proba)),
    }

    for method in methods:
        calibrated = CalibratedClassifierCV(
            estimator=build_base_estimator(y_train), method=method, cv=cv_folds
        )
        proba = cross_val_predict(calibrated, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        scores[method] = {
            "brier_score": float(brier_score_loss(y_train, proba)),
            "log_loss": float(log_loss(y_train, proba, labels=[0, 1])),
            "roc_auc": float(roc_auc_score(y_train, proba)),
        }

    best_method = min(methods, key=lambda m: scores[m]["brier_score"])
    return {
        "best_method": best_method,
        "cv_scores": scores,
        "improves_on_uncalibrated": scores[best_method]["brier_score"]
        < scores["uncalibrated"]["brier_score"],
    }


def fit_calibrated_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "isotonic",
    cv_folds: int = 5,
) -> CalibratedClassifierCV:
    """Fits a cross-validated calibrated classifier on the training matrix."""
    calibrated = CalibratedClassifierCV(
        estimator=build_base_estimator(y_train), method=method, cv=cv_folds
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def compute_calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Returns calibration quality metrics for one set of predicted probabilities."""
    return {
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "mean_predicted": float(np.mean(y_proba)),
        "observed_rate": float(np.mean(y_true)),
    }


def plot_calibration_curves(
    y_true: np.ndarray,
    proba_before: np.ndarray,
    proba_after: np.ndarray,
    n_bins: int = 10,
    label_after: str = "Calibrated",
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Saves a reliability diagram comparing probabilities before and after calibration.

    A perfectly calibrated model lies on the diagonal: among users scored at 0.3, three
    in ten actually purchase.
    """
    out = Path(output_path) if output_path else CALIBRATION_PLOT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_rel, ax_hist) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax_rel.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for proba, label, color, marker in (
        (proba_before, "Uncalibrated", "#c0504d", "o"),
        (proba_after, label_after, "#2f6f9f", "s"),
    ):
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
        brier = brier_score_loss(y_true, proba)
        ax_rel.plot(
            mean_pred,
            frac_pos,
            marker=marker,
            color=color,
            label=f"{label} (Brier {brier:.4f})",
        )

    ax_rel.set_ylabel("Observed purchase frequency")
    ax_rel.set_title("Purchase Propensity: Calibration Before and After")
    ax_rel.legend(loc="upper left")
    ax_rel.grid(alpha=0.3, linestyle=":")

    ax_hist.hist(proba_before, bins=20, alpha=0.6, color="#c0504d", label="Uncalibrated")
    ax_hist.hist(proba_after, bins=20, alpha=0.6, color="#2f6f9f", label=label_after)
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Users")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3, linestyle=":")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def compute_decile_lift(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> pl.DataFrame:
    """
    Ranks users by predicted probability and summarizes each decile.

    Decile 1 holds the highest scored users. Lift is a decile's observed purchase rate
    divided by the population rate, so a lift of 1.8 means that decile buys 1.8 times as
    often as a randomly chosen user. The cumulative capture column answers the question
    the gains chart asks: what share of all actual purchasers sits in the top N deciles?
    """
    order = np.argsort(-y_proba, kind="stable")
    y_sorted = np.asarray(y_true)[order]
    proba_sorted = np.asarray(y_proba)[order]

    total = len(y_sorted)
    total_positives = float(np.sum(y_sorted))
    overall_rate = total_positives / total if total else 0.0

    bins = np.array_split(np.arange(total), n_bins)
    rows: List[Dict[str, Any]] = []
    cumulative_positives = 0.0
    cumulative_count = 0

    for idx, bin_indices in enumerate(bins, start=1):
        if len(bin_indices) == 0:
            continue
        bin_y = y_sorted[bin_indices]
        bin_p = proba_sorted[bin_indices]
        positives = float(np.sum(bin_y))
        cumulative_positives += positives
        cumulative_count += len(bin_indices)
        decile_rate = positives / len(bin_indices)

        rows.append(
            {
                "decile": idx,
                "users": int(len(bin_indices)),
                "mean_predicted_probability": float(np.mean(bin_p)),
                "actual_purchasers": int(positives),
                "actual_purchase_rate": decile_rate,
                "lift": (decile_rate / overall_rate) if overall_rate > 0 else 0.0,
                "cumulative_users_pct": cumulative_count / total,
                "cumulative_capture_pct": (
                    cumulative_positives / total_positives if total_positives > 0 else 0.0
                ),
            }
        )

    return pl.DataFrame(rows)


def plot_lift_chart(
    lift_df: pl.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Saves a decile lift bar chart with the random baseline marked at 1.0."""
    out = Path(output_path) if output_path else LIFT_PLOT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    deciles = lift_df["decile"].to_list()
    lifts = lift_df["lift"].to_list()
    rates = lift_df["actual_purchase_rate"].to_list()

    fig, (ax_lift, ax_rate) = plt.subplots(1, 2, figsize=(13, 5))

    ax_lift.bar(deciles, lifts, color="#2f6f9f")
    ax_lift.axhline(1.0, color="#c0504d", linestyle="--", label="Random targeting")
    ax_lift.set_xlabel("Decile (1 = highest predicted probability)")
    ax_lift.set_ylabel("Lift over population rate")
    ax_lift.set_title("Decile Lift")
    ax_lift.set_xticks(deciles)
    ax_lift.legend()
    ax_lift.grid(axis="y", alpha=0.3, linestyle=":")

    ax_rate.bar(deciles, rates, color="#4f81a8")
    ax_rate.set_xlabel("Decile (1 = highest predicted probability)")
    ax_rate.set_ylabel("Observed purchase rate")
    ax_rate.set_title("Observed Purchase Rate by Decile")
    ax_rate.set_xticks(deciles)
    ax_rate.grid(axis="y", alpha=0.3, linestyle=":")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def compute_cumulative_gains(y_true: np.ndarray, y_proba: np.ndarray) -> pl.DataFrame:
    """
    Returns the full cumulative gains curve, one row per user in score order.

    This is the per-user resolution version of the decile capture column, used to draw a
    smooth gains curve and to read off capture at an arbitrary cutoff.
    """
    order = np.argsort(-np.asarray(y_proba), kind="stable")
    y_sorted = np.asarray(y_true)[order]
    total = len(y_sorted)
    total_positives = float(np.sum(y_sorted))

    cumulative_positives = np.cumsum(y_sorted).astype(float)
    population_pct = (np.arange(1, total + 1)) / total
    capture_pct = (
        cumulative_positives / total_positives if total_positives > 0 else np.zeros(total)
    )

    return pl.DataFrame(
        {
            "population_pct": population_pct,
            "capture_pct": capture_pct,
            "cumulative_purchasers": cumulative_positives,
        }
    )


def capture_at_population_fraction(gains_df: pl.DataFrame, fraction: float) -> float:
    """Returns the share of all purchasers captured in the top `fraction` of users."""
    subset = gains_df.filter(pl.col("population_pct") <= fraction)
    if subset.height == 0:
        return 0.0
    return float(subset["capture_pct"].tail(1)[0])


def plot_cumulative_gains(
    gains_df: pl.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    highlight_fractions: Tuple[float, ...] = (0.1, 0.2, 0.3),
) -> Path:
    """Saves the cumulative gains curve against the random baseline."""
    out = Path(output_path) if output_path else GAINS_PLOT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    pop = [0.0] + gains_df["population_pct"].to_list()
    cap = [0.0] + gains_df["capture_pct"].to_list()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pop, cap, color="#2f6f9f", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], "k:", label="Random targeting")

    for fraction in highlight_fractions:
        captured = capture_at_population_fraction(gains_df, fraction)
        ax.plot([fraction, fraction], [0, captured], color="#c0504d", linestyle="--", alpha=0.7)
        ax.plot([0, fraction], [captured, captured], color="#c0504d", linestyle="--", alpha=0.7)
        ax.annotate(
            f"top {fraction:.0%}: {captured:.0%}",
            xy=(fraction, captured),
            xytext=(fraction + 0.03, captured - 0.07),
            fontsize=9,
            color="#c0504d",
        )

    ax.set_xlabel("Share of users contacted, highest score first")
    ax.set_ylabel("Share of all actual purchasers captured")
    ax.set_title("Purchase Propensity: Cumulative Gains")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, linestyle=":")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run_evaluation_pipeline(
    target_col: str = PRIMARY_TARGET,
    n_bins: int = 10,
    calibrated_model_path: Optional[Union[str, Path]] = None,
    summary_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Runs the full calibration and lift analysis and writes every chart and summary.

    Returns the decile table, the gains curve, before and after calibration metrics, and
    the paths of the artifacts written.
    """
    X_train, y_train, feature_names = get_modeling_matrices(split="train", target_col=target_col)
    X_test, y_test, _ = get_modeling_matrices(split="test", target_col=target_col)

    print("Fitting uncalibrated propensity model...")
    base_model = train_propensity_model(X_train, y_train)
    proba_before = base_model.predict_proba(X_test)[:, 1]

    print("Selecting a calibration method by cross-validated Brier score on train...")
    selection = select_calibration_method(X_train, y_train)
    best_method = selection["best_method"]
    print(f"  Uncalibrated CV Brier : {selection['cv_scores']['uncalibrated']['brier_score']:.4f}")
    for method in CALIBRATION_METHODS:
        print(f"  {method:<10s} CV Brier : {selection['cv_scores'][method]['brier_score']:.4f}")
    print(f"  Selected method       : {best_method}")

    print(f"Fitting {best_method} calibrated model on the training matrix...")
    calibrated_model = fit_calibrated_model(X_train, y_train, method=best_method)
    proba_after = calibrated_model.predict_proba(X_test)[:, 1]

    metrics_before = compute_calibration_metrics(y_test, proba_before)
    metrics_after = compute_calibration_metrics(y_test, proba_after)

    print("\n--- Calibration on the Test Split ---")
    print(f"{'Metric':<18} | {'Uncalibrated':>14} | {best_method.capitalize():>14}")
    print("-" * 54)
    for key in ("brier_score", "log_loss", "roc_auc", "mean_predicted"):
        print(f"{key:<18} | {metrics_before[key]:>14.4f} | {metrics_after[key]:>14.4f}")
    print(f"{'observed_rate':<18} | {metrics_before['observed_rate']:>14.4f} | "
          f"{metrics_after['observed_rate']:>14.4f}")

    # Report plainly whether recalibration was actually warranted. Because the
    # propensity model is fitted against log loss, a proper scoring rule, it tends to
    # emerge already well calibrated, and forcing a calibrator on top can cost a little
    # accuracy rather than adding any.
    if selection["improves_on_uncalibrated"]:
        print(f"\n  Recalibration helps: {best_method} lowers cross-validated Brier score.")
    else:
        print(
            f"\n  Recalibration is not needed here: the uncalibrated model already has the "
            f"lower cross-validated Brier score\n  "
            f"({selection['cv_scores']['uncalibrated']['brier_score']:.4f} against "
            f"{selection['cv_scores'][best_method]['brier_score']:.4f} for {best_method}). "
            f"The calibrated variant is\n  still fitted and saved for comparison."
        )

    # Lift and gains use the calibrated scores, which preserve ranking while being
    # directly readable as expected purchase rates.
    lift_df = compute_decile_lift(y_test, proba_after, n_bins=n_bins)
    gains_df = compute_cumulative_gains(y_test, proba_after)

    print("\n--- Decile Lift (Test Split) ---")
    print(f"{'Decile':>6} | {'Users':>5} | {'Mean Pred':>9} | {'Actual Rate':>11} | "
          f"{'Lift':>5} | {'Cum Capture':>11}")
    print("-" * 68)
    for row in lift_df.iter_rows(named=True):
        print(
            f"{row['decile']:>6} | {row['users']:>5} | {row['mean_predicted_probability']:>9.4f} | "
            f"{row['actual_purchase_rate']:>11.4f} | {row['lift']:>5.2f} | "
            f"{row['cumulative_capture_pct']:>11.2%}"
        )

    capture_10 = capture_at_population_fraction(gains_df, 0.10)
    capture_20 = capture_at_population_fraction(gains_df, 0.20)
    capture_30 = capture_at_population_fraction(gains_df, 0.30)
    top_decile_lift = float(lift_df["lift"].head(1)[0])

    print("\n--- Business Readout ---")
    print(f"  Top 10 percent of predicted users capture {capture_10:.1%} of all purchasers.")
    print(f"  Top 20 percent of predicted users capture {capture_20:.1%} of all purchasers.")
    print(f"  Top 30 percent of predicted users capture {capture_30:.1%} of all purchasers.")
    print(f"  The top decile is {top_decile_lift:.2f} times more likely to purchase than random.")

    calib_out = Path(calibrated_model_path) if calibrated_model_path else PROPENSITY_CALIBRATED_PATH
    calib_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, calib_out)

    calibration_plot = plot_calibration_curves(
        y_test, proba_before, proba_after, label_after=f"{best_method.capitalize()} calibrated"
    )
    lift_plot = plot_lift_chart(lift_df)
    gains_plot = plot_cumulative_gains(gains_df)

    s_out = Path(summary_path) if summary_path else EVALUATION_SUMMARY_PATH
    summary = {
        "target": target_col,
        "calibration_method": best_method,
        "method_selection": selection,
        "test_metrics_uncalibrated": metrics_before,
        "test_metrics_calibrated": metrics_after,
        "top_decile_lift": top_decile_lift,
        "capture_top_10_pct": capture_10,
        "capture_top_20_pct": capture_20,
        "capture_top_30_pct": capture_30,
        "decile_table": lift_df.to_dicts(),
    }
    with open(s_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved calibrated model to  {calib_out}")
    print(f"Saved calibration plot to  {calibration_plot}")
    print(f"Saved lift chart to        {lift_plot}")
    print(f"Saved gains chart to       {gains_plot}")
    print(f"Saved summary to           {s_out}")

    return {
        "base_model": base_model,
        "calibrated_model": calibrated_model,
        "feature_names": feature_names,
        "proba_before": proba_before,
        "proba_after": proba_after,
        "y_test": y_test,
        "lift_table": lift_df,
        "gains_curve": gains_df,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "summary": summary,
        "paths": {
            "calibrated_model": calib_out,
            "calibration_plot": calibration_plot,
            "lift_plot": lift_plot,
            "gains_plot": gains_plot,
            "summary": s_out,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration, lift, and gains analysis.")
    parser.add_argument("--target", type=str, default=PRIMARY_TARGET, help="Binary target column")
    parser.add_argument("--bins", type=int, default=10, help="Number of ranking bins (deciles)")
    args = parser.parse_args()

    run_evaluation_pipeline(target_col=args.target, n_bins=args.bins)

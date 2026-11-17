"""
Group-safe probability calibration for the purchase propensity model.

The training matrix holds several rows per user (one per train reference date), so any
cross-validation that splits rows rather than users lets a calibrator see a user's score
from a model that was fitted on that same user. `CalibratedClassifierCV` cannot take
groups through its internal folds, so this module does the procedure by hand:

1. out-of-fold raw probabilities from grouped, stratified folds;
2. each candidate calibrator scored by a second grouped cross-validation over those
   out-of-fold scores (Brier score);
3. the lower Brier score wins;
4. the winning calibrator is fitted on all out-of-fold scores;
5. the base model is fitted on all training rows;
6. both are wrapped in `CalibratedPropensityModel`, a picklable two-column predictor.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from src.config import RANDOM_SEED
from src.models.propensity import DEFAULT_PARAMS, compute_scale_pos_weight

PROBABILITY_CLIP = 1e-6
CALIBRATION_METHODS: Tuple[str, ...] = ("sigmoid", "isotonic")


def build_base_estimator(y_train: np.ndarray, n_estimators: int = 50) -> lgb.LGBMClassifier:
    """
    Returns an unfitted LightGBM estimator configured like the propensity model.

    A fixed boosting round count replaces early stopping, because the same configuration
    is refitted on every cross-validation fold and on the full matrix, and the calibrator
    is only valid for scores from that exact configuration.
    """
    params = dict(DEFAULT_PARAMS)
    params["n_estimators"] = n_estimators
    params["scale_pos_weight"] = compute_scale_pos_weight(y_train)
    return lgb.LGBMClassifier(**params)


def _clipped_logit(p: np.ndarray) -> np.ndarray:
    """Returns logit(p) with p clipped away from 0 and 1, as a column vector."""
    p = np.clip(np.asarray(p, dtype=float), PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def _new_calibrator(method: str) -> Any:
    """Returns an unfitted calibrator for `method`."""
    if method == "sigmoid":
        return LogisticRegression()
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    raise ValueError(f"unknown calibration method: {method!r}")


def _fit_calibrator(method: str, raw: np.ndarray, y: np.ndarray) -> Any:
    """Fits a calibrator of `method` mapping raw positive-class scores to probabilities."""
    calibrator = _new_calibrator(method)
    if method == "sigmoid":
        calibrator.fit(_clipped_logit(raw), y)
    else:
        calibrator.fit(np.asarray(raw, dtype=float), y)
    return calibrator


def _apply_calibrator(calibrator: Any, method: str, raw: np.ndarray) -> np.ndarray:
    """Maps raw positive-class scores through a fitted calibrator, clipped to [0, 1]."""
    if method == "sigmoid":
        p = calibrator.predict_proba(_clipped_logit(raw))[:, 1]
    elif method == "isotonic":
        p = calibrator.predict(np.asarray(raw, dtype=float))
    else:
        raise ValueError(f"unknown calibration method: {method!r}")
    return np.clip(np.nan_to_num(np.asarray(p, dtype=float), nan=0.0), 0.0, 1.0)


class CalibratedPropensityModel:
    """
    A fitted base classifier followed by a fitted calibrator on its positive-class score.

    Behaves like a scikit-learn binary classifier for scoring: `predict_proba` returns an
    (n, 2) matrix whose rows sum to 1. Defined at module level so joblib can pickle it.
    """

    def __init__(self, base_model: Any, calibrator: Any, method: str) -> None:
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method
        self.classes_ = np.array([0, 1])

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Returns the base model's uncalibrated positive-class probability."""
        return np.asarray(self.base_model.predict_proba(X))[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns calibrated class probabilities, shape (n, 2)."""
        p = _apply_calibrator(self.calibrator, self.method, self.predict_raw(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns hard labels at `threshold` on the calibrated probability."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def grouped_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns the (fit_idx, heldout_idx) pairs of a shuffled StratifiedGroupKFold.

    Every group is held out exactly once and never appears on both sides of a fold.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def _score(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Brier score, log loss, and ROC-AUC of probabilities `p` against `y`."""
    scores = {
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP),
                                   labels=[0, 1])),
    }
    scores["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return scores


def fit_grouped_calibration(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    methods: Sequence[str] = ("sigmoid", "isotonic"),
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> Tuple[CalibratedPropensityModel, Dict[str, Any]]:
    """
    Fits a calibrated propensity model without letting any user cross a fold boundary.

    Returns the wrapper and a dict with `best_method`, `cv_scores` (one entry per method),
    `uncalibrated_cv_scores` (the raw out-of-fold scores judged the same way),
    `oof_probabilities` (raw out-of-fold scores, one per row),
    `oof_calibrated_probabilities` (each row calibrated by a calibrator that never saw
    its group), `improves_on_uncalibrated`, and `folds` (the grouped fold indices).
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    methods = tuple(methods)
    if not methods:
        raise ValueError("methods must name at least one calibration method")

    folds = grouped_folds(y, groups, n_splits=n_splits, random_state=random_state)

    # (1) Out-of-fold raw scores: each row is scored by a model fitted without its user.
    oof = cross_val_predict(
        build_base_estimator(y), X, y, cv=folds, method="predict_proba"
    )[:, 1]

    # (2) Score each calibrator with a second grouped cross-validation over the OOF scores.
    cv_scores: Dict[str, Dict[str, float]] = {}
    cross_fitted: Dict[str, np.ndarray] = {}
    for method in methods:
        calibrated = np.empty(len(y), dtype=float)
        for fit_idx, held_idx in folds:
            calibrator = _fit_calibrator(method, oof[fit_idx], y[fit_idx])
            calibrated[held_idx] = _apply_calibrator(calibrator, method, oof[held_idx])
        cross_fitted[method] = calibrated
        cv_scores[method] = _score(y, calibrated)

    # (3) The lower cross-validated Brier score wins.
    best_method = min(methods, key=lambda m: cv_scores[m]["brier_score"])

    # (4) The chosen calibrator is fitted on every out-of-fold score.
    calibrator = _fit_calibrator(best_method, oof, y)

    # (5) The base model, same configuration, is fitted on every training row.
    base_model = build_base_estimator(y)
    base_model.fit(X, y)

    uncalibrated = _score(y, oof)
    info: Dict[str, Any] = {
        "best_method": best_method,
        "cv_scores": cv_scores,
        "uncalibrated_cv_scores": uncalibrated,
        "improves_on_uncalibrated": bool(
            cv_scores[best_method]["brier_score"] < uncalibrated["brier_score"]
        ),
        "oof_probabilities": oof,
        "oof_calibrated_probabilities": cross_fitted[best_method],
        "folds": folds,
    }
    return CalibratedPropensityModel(base_model, calibrator, best_method), info

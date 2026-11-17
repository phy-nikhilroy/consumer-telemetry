"""
Tests for group-safe probability calibration.

The calibrator must be fitted on out-of-fold scores produced with grouped folds, so no
user ever appears on both sides of a split, and the wrapper must behave as a plain
two-column probability model that survives pickling.
"""

from __future__ import annotations

import inspect
import json

import joblib
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings, strategies as st
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.calibration import CalibratedPropensityModel, fit_grouped_calibration


def _data(seed: int = 0, n: int = 240, n_groups: int = 48):
    """Builds a seeded binary problem where each group owns several rows."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logit = 1.2 * X[:, 0] - 0.8 * X[:, 1]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    groups = np.repeat(np.arange(n_groups), n // n_groups)
    return X, y, groups


def _logit(v):
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return np.log(v / (1 - v))


class _FixedBase:
    """A fitted stand-in whose positive probability is the first column of X."""

    def predict_proba(self, X):
        p = np.asarray(X, dtype=float)[:, 0]
        return np.column_stack([1.0 - p, p])


class _Recorder:
    """
    Records every LightGBM fit and predict call. The group id rides in the last column
    of X as an extra feature, so each call logs which groups it saw.
    """

    def __init__(self, monkeypatch):
        import lightgbm as lgb

        self.fits: list = []
        self.predicts: list = []
        orig_fit, orig_pp = lgb.LGBMClassifier.fit, lgb.LGBMClassifier.predict_proba
        rec = self

        def fit(est, X, y, *a, **k):
            rec.fits.append((np.unique(np.asarray(X)[:, -1]), len(X)))
            return orig_fit(est, X, y, *a, **k)

        def predict_proba(est, X, *a, **k):
            rec.predicts.append(np.unique(np.asarray(X)[:, -1]))
            return orig_pp(est, X, *a, **k)

        monkeypatch.setattr(lgb.LGBMClassifier, "fit", fit)
        monkeypatch.setattr(lgb.LGBMClassifier, "predict_proba", predict_proba)

    def fold_fits(self, n_rows):
        return [fg for fg, n in self.fits if n < n_rows]


def _fit(methods=("sigmoid", "isotonic"), seed=0, n_splits=4, X=None, y=None, g=None):
    if X is None:
        X, y, g = _data(seed)
    return fit_grouped_calibration(X, y, g, methods=methods, n_splits=n_splits), (X, y, g)


# ---------------------------------------------------------------- signature and names


def test_fit_grouped_calibration_signature_matches_spec():
    """Verifies the documented parameter names, order and defaults."""
    from src.config import RANDOM_SEED

    params = inspect.signature(fit_grouped_calibration).parameters
    assert list(params)[:3] == ["X", "y", "groups"]
    assert params["methods"].default == ("sigmoid", "isotonic")
    assert params["n_splits"].default == 5
    assert params["random_state"].default == RANDOM_SEED


def test_wrapper_constructor_signature():
    """Verifies the wrapper takes (base_model, calibrator, method)."""
    assert list(inspect.signature(CalibratedPropensityModel.__init__).parameters)[1:] == [
        "base_model",
        "calibrator",
        "method",
    ]


# ---------------------------------------------------------------- wrapper behaviour


def test_predict_proba_rows_sum_to_one_and_lie_in_unit_interval():
    """Verifies the fitted wrapper returns an (n, 2) valid probability matrix."""
    for method in ("sigmoid", "isotonic"):
        (model, _), (X, _, _) = _fit(methods=(method,))
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert proba.min() >= 0.0 and proba.max() <= 1.0
        assert np.isfinite(proba).all()


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=1, max_size=40))
def test_predict_proba_valid_for_any_raw_score(raw):
    """Verifies sigmoid and isotonic wrappers stay valid for any raw score in [0, 1]."""
    rng = np.random.default_rng(3)
    scores = rng.random(60)
    labels = (rng.random(60) < scores).astype(int)
    lr = LogisticRegression().fit(_logit(scores).reshape(-1, 1), labels)
    iso = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
    X = np.array(raw, dtype=float).reshape(-1, 1)
    for cal, method in ((lr, "sigmoid"), (iso, "isotonic")):
        proba = CalibratedPropensityModel(_FixedBase(), cal, method).predict_proba(X)
        assert proba.shape == (len(raw), 2)
        assert np.all(np.isfinite(proba))
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_sigmoid_wrapper_uses_clipped_logit_of_raw_probability():
    """Verifies the sigmoid path is LogisticRegression on logit(clip(p)), finite at 0 and 1."""
    rng = np.random.default_rng(5)
    p = rng.random(80)
    y = (rng.random(80) < p).astype(int)
    lr = LogisticRegression().fit(_logit(p).reshape(-1, 1), y)
    model = CalibratedPropensityModel(_FixedBase(), lr, "sigmoid")

    probe = np.array([[0.0], [1e-9], [0.3], [0.5], [0.9], [1.0]])
    got = model.predict_proba(probe)[:, 1]
    want = lr.predict_proba(_logit(probe[:, 0]).reshape(-1, 1))[:, 1]
    assert np.isfinite(got).all()
    assert np.allclose(got, want)


def test_isotonic_wrapper_output_is_monotone_non_decreasing_in_raw_score():
    """Verifies sweeping the raw score upward never decreases the isotonic output."""
    rng = np.random.default_rng(7)
    p = rng.random(200)
    y = (rng.random(200) < p**2).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    model = CalibratedPropensityModel(_FixedBase(), iso, "isotonic")
    grid = np.linspace(0.0, 1.0, 201).reshape(-1, 1)
    out = model.predict_proba(grid)[:, 1]
    assert np.all(np.diff(out) >= -1e-12)


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=2, max_size=60))
def test_isotonic_monotone_for_arbitrary_raw_scores(raw):
    """Verifies monotonicity on arbitrary raw scores, including ones outside the training range."""
    rng = np.random.default_rng(11)
    p = rng.random(100) * 0.6 + 0.2
    y = (rng.random(100) < p).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    model = CalibratedPropensityModel(_FixedBase(), iso, "isotonic")
    out = model.predict_proba(np.sort(np.array(raw)).reshape(-1, 1))[:, 1]
    assert np.all(np.diff(out) >= -1e-12)


def test_fitted_isotonic_model_is_monotone_in_base_score():
    """Verifies the model from fit_grouped_calibration is monotone in its own raw score."""
    (model, _), (X, _, _) = _fit(methods=("isotonic",))
    raw = model.base_model.predict_proba(X)[:, 1]
    order = np.argsort(raw, kind="stable")
    cal = model.predict_proba(X)[:, 1][order]
    assert np.all(np.diff(cal) >= -1e-12)


def test_wrapper_survives_joblib_roundtrip(tmp_path):
    """Verifies dump and load reproduce identical predictions and metadata."""
    for method in ("sigmoid", "isotonic"):
        (model, _), (X, _, _) = _fit(methods=(method,))
        path = tmp_path / f"{method}.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)
        assert isinstance(loaded, CalibratedPropensityModel)
        assert loaded.method == method
        assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))


def test_wrapper_exposes_method_attribute():
    """Verifies the chosen method name is stored on the wrapper."""
    iso = IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.9], [0, 1])
    assert CalibratedPropensityModel(_FixedBase(), iso, "isotonic").method == "isotonic"


# ---------------------------------------------------------------- fit_grouped_calibration


def test_info_dict_has_spec_keys_and_shapes():
    """Verifies best_method, cv_scores and oof_probabilities are returned and consistent."""
    (model, info), (_, y, _) = _fit()
    assert {"best_method", "cv_scores", "oof_probabilities"} <= set(info)
    assert set(info["cv_scores"]) >= {"sigmoid", "isotonic"}
    oof = np.asarray(info["oof_probabilities"])
    assert oof.shape == (len(y),)
    assert np.isfinite(oof).all() and oof.min() >= 0.0 and oof.max() <= 1.0
    assert model.method == info["best_method"]


def test_best_method_has_lowest_brier_score():
    """Verifies the selection is the argmin of the cross-validated Brier scores."""
    (_, info), _ = _fit()

    def brier(v):
        return v["brier_score"] if isinstance(v, dict) else float(v)

    scores = {m: brier(info["cv_scores"][m]) for m in ("sigmoid", "isotonic")}
    assert scores[info["best_method"]] == min(scores.values())
    assert all(0.0 <= s <= 1.0 for s in scores.values())


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_single_method_is_always_selected(method):
    """Verifies restricting methods forces that method into the wrapper."""
    (model, info), _ = _fit(methods=(method,))
    assert info["best_method"] == method
    assert model.method == method
    assert set(info["cv_scores"]) == {method}


def test_calibrator_type_matches_method():
    """Verifies sigmoid uses LogisticRegression and isotonic uses clipping IsotonicRegression."""
    (sig, _), _ = _fit(methods=("sigmoid",))
    (iso, _), _ = _fit(methods=("isotonic",))
    assert isinstance(sig.calibrator, LogisticRegression)
    assert isinstance(iso.calibrator, IsotonicRegression)
    assert iso.calibrator.out_of_bounds == "clip"


def test_fit_is_deterministic_for_a_fixed_seed():
    """Verifies the same inputs and random_state give identical out-of-fold scores and predictions."""
    X, y, g = _data(2)
    (m1, i1), _ = _fit(X=X, y=y, g=g)
    (m2, i2), _ = _fit(X=X, y=y, g=g)
    assert np.array_equal(i1["oof_probabilities"], i2["oof_probabilities"])
    assert i1["best_method"] == i2["best_method"]
    assert np.array_equal(m1.predict_proba(X), m2.predict_proba(X))


def test_grouped_folds_never_share_a_user_between_fit_and_heldout(monkeypatch):
    """Verifies every scored fold batch is disjoint from the groups its model was fitted on."""
    rec = _Recorder(monkeypatch)
    X, y, g = _data(4)
    n_groups = len(np.unique(g))
    _fit(methods=("sigmoid", "isotonic"), n_splits=4, X=np.column_stack([X, g]), y=y, g=g)

    fold_fits = rec.fold_fits(len(y))
    assert len(fold_fits) == 4, "expected one base fit per fold"
    held_batches = [s for s in rec.predicts if len(s) < n_groups]
    assert len(held_batches) >= 4
    for batch in held_batches:
        assert any(np.intersect1d(fg, batch).size == 0 for fg in fold_fits)
    # Pair them exactly: the held-out groups of fold k are all groups not fitted in fold k.
    for fg in fold_fits:
        held = np.setdiff1d(np.unique(g), fg)
        assert len(held) > 0
        assert any(np.array_equal(np.sort(b), held) for b in held_batches)


def test_every_user_is_held_out_exactly_once(monkeypatch):
    """Verifies the held-out groups of the folds partition the population."""
    rec = _Recorder(monkeypatch)
    X, y, g = _data(5)
    _fit(methods=("sigmoid",), n_splits=4, X=np.column_stack([X, g]), y=y, g=g)
    held = np.concatenate([np.setdiff1d(np.unique(g), fg) for fg in rec.fold_fits(len(y))])
    assert len(held) == len(np.unique(g))
    assert len(np.unique(held)) == len(np.unique(g))


def test_final_base_model_is_fitted_on_all_training_rows(monkeypatch):
    """Verifies one fit per fold plus exactly one fit on every row."""
    rec = _Recorder(monkeypatch)
    X, y, g = _data(6)
    _fit(methods=("sigmoid", "isotonic"), n_splits=4, X=np.column_stack([X, g]), y=y, g=g)
    sizes = [n for _, n in rec.fits]
    assert sizes.count(len(y)) == 1
    assert len(sizes) == 4 + 1


def test_oof_probabilities_are_not_in_sample_scores():
    """Verifies out-of-fold scores differ from the final model's in-sample raw scores."""
    (model, info), (X, _, _) = _fit(methods=("sigmoid",))
    raw_in_sample = model.base_model.predict_proba(X)[:, 1]
    assert not np.allclose(np.asarray(info["oof_probabilities"]), raw_in_sample, atol=1e-12)


def test_duplicated_rows_within_a_group_stay_in_one_fold(monkeypatch):
    """Verifies groups whose rows are identical are never split across fit and held-out parts."""
    rec = _Recorder(monkeypatch)
    rng = np.random.default_rng(9)
    g = np.repeat(np.arange(40), 6)
    X = np.repeat(rng.normal(size=(40, 3)), 6, axis=0)
    y = np.repeat((rng.random(40) < 0.5).astype(int), 6)
    _fit(methods=("sigmoid",), n_splits=4, X=np.column_stack([X, g]), y=y, g=g)
    fold_fits = rec.fold_fits(len(y))
    assert len(fold_fits) == 4
    for fg in fold_fits:
        # A fold fitted on n rows holds out a whole number of groups of 6 rows.
        assert len(fg) < 40


# ---------------------------------------------------------------- evaluation integration


def test_evaluation_no_longer_uses_calibrated_classifier_cv():
    """Verifies evaluation.py is wired to fit_grouped_calibration, not CalibratedClassifierCV."""
    import src.models.evaluation as ev

    source = inspect.getsource(ev)
    assert "fit_grouped_calibration" in source
    assert "CalibratedClassifierCV(" not in source


@pytest.fixture(scope="module")
def evaluation_run(tmp_path_factory):
    """Runs the evaluation pipeline on a small synthetic feature store in a temp directory."""
    from src.models.evaluation import run_evaluation_pipeline

    root = tmp_path_factory.mktemp("calib_eval")
    features_dir, artifacts_dir = root / "features", root / "artifacts"
    features_dir.mkdir()
    rng = np.random.default_rng(21)

    def frame(n, offset):
        X = rng.normal(size=(n, 3))
        p = 1.0 / (1.0 + np.exp(-(1.5 * X[:, 0] - 0.5)))
        return pl.DataFrame(
            {
                "user_id": [f"u{offset + i // 3}" for i in range(n)],
                "f1": X[:, 0],
                "f2": X[:, 1],
                "f3": X[:, 2],
                "will_purchase_next_7d": (rng.random(n) < p).astype(int),
            }
        )

    frame(300, 0).write_parquet(features_dir / "train_features.parquet")
    frame(120, 1000).write_parquet(features_dir / "test_features.parquet")
    (features_dir / "selected_features.json").write_text(
        json.dumps({"selected_features": ["f1", "f2", "f3"]})
    )
    result = run_evaluation_pipeline(features_dir=features_dir, artifacts_dir=artifacts_dir)
    return result, artifacts_dir


def test_saved_artifact_is_a_calibrated_propensity_model(evaluation_run):
    """Verifies the file at the calibrated path loads as CalibratedPropensityModel."""
    result, artifacts_dir = evaluation_run
    loaded = joblib.load(artifacts_dir / "propensity_lgbm_calibrated.joblib")
    assert isinstance(loaded, CalibratedPropensityModel)
    assert isinstance(result["calibrated_model"], CalibratedPropensityModel)


def test_summary_contains_risk_bands(evaluation_run):
    """Verifies risk_bands has exactly low_below and high_at_or_above, ordered and in [0, 1]."""
    _, artifacts_dir = evaluation_run
    summary = json.loads((artifacts_dir / "propensity_evaluation_summary.json").read_text())
    bands = summary["risk_bands"]
    assert set(bands) == {"low_below", "high_at_or_above"}
    assert 0.0 <= bands["low_below"] <= bands["high_at_or_above"] <= 1.0


def test_risk_bands_split_calibrated_scores_into_roughly_thirds(evaluation_run):
    """Verifies the bands are the 33rd and 67th percentiles, so each tail holds about a third."""
    result, artifacts_dir = evaluation_run
    bands = json.loads(
        (artifacts_dir / "propensity_evaluation_summary.json").read_text()
    )["risk_bands"]
    proba = np.asarray(result["proba_after"])
    assert 0.10 <= float(np.mean(proba < bands["low_below"])) <= 0.55
    assert 0.10 <= float(np.mean(proba >= bands["high_at_or_above"])) <= 0.55

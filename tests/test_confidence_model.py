"""Tests for the learned extraction confidence model."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.ingestion import confidence_model as cm
from src.ingestion.ner_extractor import BRAND_SOURCE_LEADING_MATCH, BRAND_SOURCE_LEADING_TOKEN, BRAND_SOURCE_MATCH
from src.ingestion.pipeline import calculate_extraction_confidence, calculate_hand_weighted_confidence

RECORD = {
    "brand": "Samsung", "brand_source": BRAND_SOURCE_LEADING_TOKEN, "product_title": "Galaxy S6 Case Red",
    "price": 9.99, "price_verified": True, "currency": "USD", "rating": 4.2,
    "category_l2": "Smartphones", "category_similarity": 0.4, "category_probability": 0.7, "category_model_margin": 0.3,
}


def test_feature_vector_order_and_length():
    vec = cm.feature_vector(RECORD)
    assert len(vec) == len(cm.FEATURE_NAMES) == 10
    assert cm.FEATURE_NAMES[:3] == ("brand_leading_match", "brand_leading_token", "brand_mid_match")
    assert vec == [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3, 4.0]


def test_brand_source_one_hot_and_empty_record():
    for source, idx in ((BRAND_SOURCE_LEADING_MATCH, 0), (BRAND_SOURCE_LEADING_TOKEN, 1), (BRAND_SOURCE_MATCH, 2)):
        assert cm.feature_vector({**RECORD, "brand_source": source})[:3][idx] == 1.0
    assert cm.feature_vector({"brand_source": BRAND_SOURCE_MATCH})[:3] == [0.0, 0.0, 0.0]
    assert cm.feature_vector({}) == [0.0] * 10


def _tiny_artifact(path):
    rng = np.random.default_rng(0)
    x = np.asarray([cm.feature_vector(RECORD)] * 20 + [cm.feature_vector({})] * 20) + rng.normal(0, 0.01, (40, 10))
    y = np.asarray([1] * 20 + [0] * 20)
    joblib.dump({"model": LogisticRegression(max_iter=2000).fit(x, y)}, path)


def test_probabilities_in_unit_interval(tmp_path):
    path = tmp_path / "c.joblib"
    _tiny_artifact(path)
    for rec in (RECORD, {}, {**RECORD, "title_token_count": 500, "product_title": "w " * 500}):
        p = cm.predict_confidence(rec, path)
        assert 0.0 <= p <= 1.0
    assert cm.predict_confidence(RECORD, path) > cm.predict_confidence({"brand": None, "price": 1.0}, path)


def test_ece_of_perfectly_calibrated_input_is_small():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert cm.expected_calibration_error(p, y) < 0.02


def test_ece_detects_miscalibration_and_empty():
    assert cm.expected_calibration_error([0.9] * 10, [0] * 10) > 0.85
    assert cm.expected_calibration_error([], []) == 0.0
    assert cm.expected_calibration_error([1.0, 1.0], [1, 1]) == 0.0


def test_auc_score_and_single_class():
    assert cm.auc_score([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1]) == 1.0
    assert cm.auc_score([0.1, 0.9], [1, 1]) is None


def test_fallback_to_hand_weights_when_artifact_absent(tmp_path):
    missing = tmp_path / "absent.joblib"
    assert cm.predict_confidence(RECORD, missing) is None
    assert calculate_extraction_confidence(RECORD, missing) == calculate_hand_weighted_confidence(RECORD)


def test_learned_score_used_when_artifact_present(tmp_path):
    path = tmp_path / "c.joblib"
    _tiny_artifact(path)
    assert calculate_extraction_confidence(RECORD, path) == cm.predict_confidence(RECORD, path)
    assert calculate_extraction_confidence({}, path) == 0.0


def test_record_label_rules():
    row = {"true_brand": "Samsung", "true_price": 100.0, "true_label": "Electronics > Cell Phones"}
    ok = {"brand": "samsung", "price": 100.5, "full_category": "Electronics > Cell Phones"}
    assert cm.record_label(ok, row) == 1
    assert cm.record_label({**ok, "price": 102.0}, row) == 0
    assert cm.record_label({**ok, "brand": "Apple"}, row) == 0
    assert cm.record_label({**ok, "full_category": "Electronics > Other"}, row) == 0
    assert cm.record_label(ok, {**row, "true_label": None}) == 0
    assert cm.record_label({**ok, "price": None}, row) == 0


def test_fallback_features_use_model_probability_not_prototype_similarity():
    from src.ingestion.pipeline import model_probability_and_margin

    fallback = {"method": "prototype_fallback", "confidence": 0.43, "margin": 0.05,
                "model_assigned_probability": 0.31, "model_margin": 0.12}
    prob, margin = model_probability_and_margin(fallback)
    assert (prob, margin) == (0.31, 0.12)
    vec = cm.feature_vector({**RECORD, "category_similarity": 0.43,
                             "category_probability": prob, "category_model_margin": margin})
    assert vec[7] == 0.31 and vec[8] == 0.12
    assert model_probability_and_margin({"method": "prototype", "confidence": 0.5}) == (0.0, 0.0)
    sup = {"method": "supervised", "confidence": 0.9, "margin": 0.6, "model_assigned_probability": 0.9}
    assert model_probability_and_margin(sup) == (0.9, 0.6)


def test_assigned_probability_from_stub_probabilities():
    from tests.test_taxonomy_classifier import LABELS, PROTOTYPE_TO_AMAZON, _supervised

    clf = _supervised([0.55, 0.45], l2="Smartphones")
    res = clf.classify_product("Some phone", brand="Acme")
    l1, l2 = PROTOTYPE_TO_AMAZON["Smartphones"]
    idx = LABELS.index(f"{l1} > {l2}")
    assert res["method"] == "prototype_fallback"
    assert res["model_assigned_probability"] == [0.55, 0.45][idx]
    assert res["model_margin"] == 0.1
    unmapped = _supervised([0.55, 0.45], l2="Footwear").classify_product("Shoe", brand="Acme")
    assert unmapped["model_assigned_probability"] == 0.0
    assert _supervised([0.1, 0.9]).classify_product("Lens")["model_assigned_probability"] == 0.9
    assert _supervised([0.5, 0.5], abstained=True).classify_product("zzz")["model_assigned_probability"] == 0.0

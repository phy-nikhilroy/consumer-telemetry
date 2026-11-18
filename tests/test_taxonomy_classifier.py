"""
Tests for taxonomy classification.

The first group exercises the v1.0 prototype classifier directly; it is the fallback of the
supervised model and the whole classifier when no artifact exists. The second group covers
the supervised model, its abstention, and the mapped fallback, with stand-in components.
"""

import numpy as np
import pytest

from src.ingestion import taxonomy_classifier as tc
from src.ingestion.taxonomy_classifier import (
    METHOD_FALLBACK,
    METHOD_SUPERVISED,
    PROTOTYPE_TO_AMAZON,
    SupervisedTaxonomyClassifier,
)


def classify_product(product_title, brand=None, min_similarity=None):
    """The prototype classifier, as `classify_product` behaved in v1.0."""
    return tc._prototype_classifier().classify_product(product_title, brand, min_similarity=min_similarity)


def test_classify_smartphone():
    res = classify_product("iPhone 15 Pro Max 256GB", brand="Apple")
    assert res["category_l1"] == "Electronics"
    assert res["category_l2"] == "Smartphones"
    assert "Smartphones" in res["full_category"]
    assert res["confidence"] > 0.25


def test_classify_audio_headphones():
    res = classify_product("Airdopes 141 True Wireless Earbuds", brand="boAt")
    assert res["category_l1"] == "Electronics"
    assert res["category_l2"] == "Audio"
    assert "Headphones" in res["full_category"]
    assert res["confidence"] > 0.50


def test_classify_smartwatch():
    res = classify_product("Galaxy Watch 6 Bluetooth 44mm", brand="Samsung")
    assert res["category_l1"] == "Electronics"
    assert res["category_l2"] == "Wearables"
    assert "Smartwatches" in res["full_category"]
    assert res["confidence"] > 0.35


def test_classify_footwear():
    res = classify_product("Air Monarch IV Cross Training Shoes", brand="Nike")
    assert res["category_l1"] == "Apparel & Accessories"
    assert res["category_l2"] == "Footwear"
    assert "Shoes" in res["full_category"]
    assert res["confidence"] > 0.50


def test_classify_storage():
    res = classify_product("Extreme Portable SSD 1TB", brand="SanDisk")
    assert res["category_l1"] == "Electronics"
    assert res["category_l2"] == "Storage"
    assert "Storage" in res["full_category"]
    assert res["confidence"] > 0.40


def test_classify_accessories():
    res = classify_product("MX Master 3S Wireless Performance Mouse", brand="Logitech")
    assert res["category_l1"] == "Electronics"
    assert res["category_l2"] == "Accessories"
    assert res["confidence"] > 0.35


def test_classify_empty():
    res = classify_product("", brand="")
    assert res["category_l1"] == "Unknown"
    assert res["confidence"] == 0.0


from src.config import TAXONOMY_MIN_SIMILARITY


def test_classifier_abstains_below_similarity_floor():
    """
    Verifies a product unlike every taxonomy description is labelled Unknown.

    Nearest-neighbour search over a closed label set always returns something, which
    once put a quarter of real phone accessories in categories such as Footwear.
    """
    res = classify_product("Ryleigh Name on Geometric Red Pink Purple Black PopGrip")
    assert res["confidence"] < TAXONOMY_MIN_SIMILARITY
    assert res["abstained"] is True
    assert res["category_l1"] == "Unknown"
    assert res["category_l2"] == "Unknown"
    assert res["nearest_category"] is not None


def test_classifier_floor_is_configurable():
    """Verifies the floor can be overridden per call, and reports the runner-up margin."""
    strict = classify_product("Extreme Portable SSD 1TB", brand="SanDisk", min_similarity=0.99)
    assert strict["abstained"] is True
    assert "Storage" in strict["nearest_category"]

    lenient = classify_product("Extreme Portable SSD 1TB", brand="SanDisk", min_similarity=0.0)
    assert lenient["abstained"] is False
    assert lenient["category_l2"] == "Storage"
    assert lenient["margin"] >= 0.0


class _FixedModel:
    """Returns preset probabilities for the two supervised labels."""

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=float)

    def predict_proba(self, x):
        return np.tile(self.probs, (len(x), 1))


class _StubEncoder:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        return np.ones((len(texts), 4), dtype=np.float32)


class _StubPrototype:
    """Stands in for the prototype classifier: fixed answer, stub encoder."""

    def __init__(self, l2="Smartphones", abstained=False):
        self.model = _StubEncoder()
        self.l2 = l2
        self.abstained = abstained
        self.calls = 0

    def classify_product(self, product_title, brand=None, min_similarity=None):
        self.calls += 1
        return {"category_l1": "Electronics", "category_l2": self.l2, "full_category": "x",
                "confidence": 0.3, "margin": 0.1, "abstained": self.abstained, "nearest_category": "x"}


LABELS = ["Cell Phones & Accessories > Cell Phones", "Electronics > Camera & Photo"]


def _supervised(probs, tau=0.6, **proto):
    artifact = {"model": _FixedModel(probs), "labels": LABELS, "tau": tau,
                "encoder_name": "stub", "trained_on_rows": 10}
    return SupervisedTaxonomyClassifier(artifact, prototype=_StubPrototype(**proto))


def test_supervised_confident_answer_uses_model_label():
    clf = _supervised([0.1, 0.9])
    res = clf.classify_product("Some zoom lens", brand="Canon")
    assert (res["category_l1"], res["category_l2"]) == ("Electronics", "Camera & Photo")
    assert res["confidence"] == pytest.approx(0.9)
    assert res["margin"] == pytest.approx(0.8)
    assert res["method"] == METHOD_SUPERVISED
    assert res["abstained"] is False and res["model_abstained"] is False
    assert clf.prototype.calls == 0


def test_supervised_below_tau_falls_back_to_mapped_prototype():
    clf = _supervised([0.55, 0.45], l2="Footwear")
    res = clf.classify_product("Smash v2", brand="Puma")
    assert (res["category_l1"], res["category_l2"]) == PROTOTYPE_TO_AMAZON["Footwear"]
    assert res["method"] == METHOD_FALLBACK
    assert res["model_abstained"] is True
    assert res["model_label"] == LABELS[0]
    assert res["abstained"] is False


def test_fallback_that_abstains_is_unknown():
    clf = _supervised([0.5, 0.5], abstained=True)
    res = clf.classify_product("qwerty", brand=None)
    assert res["category_l2"] == "Unknown" and res["abstained"] is True
    assert res["method"] == METHOD_FALLBACK


@pytest.mark.parametrize("title", ["", "   "])
def test_empty_title_abstains(title):
    clf = _supervised([0.0, 1.0])
    res = clf.classify_product(title, brand="Apple")
    assert res["abstained"] is True and res["category_l2"] == "Unknown"
    assert clf.prototype.calls == 0


def test_every_prototype_category_is_mapped():
    assert {r["category_l2"] for r in tc.TAXONOMY_RECORDS} <= set(PROTOTYPE_TO_AMAZON)


def test_module_falls_back_to_prototype_without_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "TAXONOMY_ARTIFACT_PATH", tmp_path / "missing.joblib")
    tc.reset_classifiers()
    try:
        res = tc.classify_product("iPhone 15 Pro Max 256GB", brand="Apple")
        assert res["method"] == "prototype"
        assert res["category_l2"] == "Smartphones"
    finally:
        tc.reset_classifiers()


def test_committed_artifact_loads_with_expected_keys():
    artifact = tc.load_taxonomy_artifact()
    if artifact is None:
        pytest.skip("taxonomy artifact not trained")
    assert set(artifact) == {"model", "labels", "tau", "encoder_name", "trained_on_rows"}
    assert 0.0 < artifact["tau"] <= 0.95
    assert len(artifact["labels"]) == len(artifact["model"].classes_)


class _RecordingModel(_FixedModel):
    """Fixed probabilities; records how many rows it was asked about."""

    def __init__(self, probs):
        super().__init__(probs)
        self.seen = 0

    def predict_proba(self, x):
        self.seen += len(x)
        return super().predict_proba(x)


def test_model_text_feeds_the_model_and_the_fallback_keeps_the_title(monkeypatch):
    queries = []
    clf = _supervised([0.55, 0.45], l2="Footwear")
    original = clf.predict_proba

    def spy(qs):
        queries.extend(qs)
        return original(qs)

    monkeypatch.setattr(clf, "predict_proba", spy)
    titles = []
    proto = clf.prototype
    real = proto.classify_product
    proto.classify_product = lambda t, b=None, min_similarity=None: titles.append((t, b)) or real(t, b, min_similarity)
    res = clf.classify_product("Smash v2", brand="Puma", model_text="Puma Smash v2 Leather Sneaker White In stock")
    assert queries == ["Puma Smash v2 Leather Sneaker White In stock"]
    assert titles == [("Smash v2", "Puma")]
    assert res["method"] == METHOD_FALLBACK


def test_blank_model_text_falls_back_to_brand_and_title(monkeypatch):
    queries = []
    clf = _supervised([0.1, 0.9])
    original = clf.predict_proba
    monkeypatch.setattr(clf, "predict_proba", lambda qs: queries.extend(qs) or original(qs))
    clf.classify_product("Zoom lens", brand="Canon", model_text="   ")
    assert queries == ["Canon Zoom lens"]

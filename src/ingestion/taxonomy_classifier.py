"""
Product taxonomy classification.

Two classifiers live here. The supervised one (v1.2) is a logistic regression over title
embeddings, trained on real Amazon listings by `src/ingestion/taxonomy_train.py` and saved at
`TAXONOMY_ARTIFACT_PATH`; it abstains when its top probability is below the tuned `tau`. The
prototype one (v1.0) matches a title against hand-written category descriptions with FAISS
and abstains below a similarity floor.

`classify_product` uses the supervised model when its artifact exists. When that model
abstains on a non-empty title, the prototype classifier answers instead, its category mapped
into the same Amazon label space (`PROTOTYPE_TO_AMAZON`), so short catalog titles the real
listings never resemble still get a consistent category. The result records which path
answered in `method`, and the supervised model's own verdict in `model_abstained`, so the
held-out benchmark can score the model alone. Without the artifact, the prototype classifier
answers exactly as in v1.0.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import TAXONOMY_ARTIFACT_PATH, TAXONOMY_MIN_SIMILARITY

# Mapping from representative taxonomy descriptions to structured L1/L2 and full hierarchy
# Aligned with Google Product Taxonomy standards and e-commerce categories
TAXONOMY_RECORDS = [
    {
        "category_l1": "Electronics",
        "category_l2": "Smartphones",
        "full_category": "Electronics > Communications > Telephony > Mobile Phones > Smartphones",
        "description": "Smartphones, mobile phone devices, Apple iPhone, Samsung Galaxy S and Galaxy M phones, Google Pixel, OnePlus Android mobile phones, 5G cellular handsets."
    },
    {
        "category_l1": "Electronics",
        "category_l2": "Audio",
        "full_category": "Electronics > Audio > Headphones & Earphones",
        "description": "Audio headphones, true wireless earbuds, boAt Rockerz, Airdopes, Sony WH-1000XM wireless industry leading noise cancelling over-ear headphones, Apple AirPods, Bluetooth earphones."
    },
    {
        "category_l1": "Electronics",
        "category_l2": "Wearables",
        "full_category": "Electronics > Wearables > Smartwatches",
        "description": "Smartwatches and fitness trackers worn on the wrist, Apple Watch Series, Noise ColorFit smartwatch, health tracking wristbands."
    },
    {
        "category_l1": "Apparel & Accessories",
        "category_l2": "Footwear",
        "full_category": "Apparel & Accessories > Shoes > Athletic Shoes & Sneakers",
        "description": "Footwear, athletic shoes, sneakers, Nike Air Monarch, Adidas Ultraboost running shoes, Puma Smash sneakers, sports trainers."
    },
    {
        "category_l1": "Electronics",
        "category_l2": "Storage",
        "full_category": "Electronics > Computers > Storage > External Solid State Drives & Flash Memory",
        "description": "Computer storage, SanDisk Extreme Portable SSD, external hard drives, USB flash drives, high-speed memory cards."
    },
    {
        "category_l1": "Electronics",
        "category_l2": "Accessories",
        "full_category": "Electronics > Computers > Computer Accessories > Keyboards & Mice",
        "description": "Computer accessories and input peripherals, Logitech MX Master wireless mouse, keyboards, ergonomic desk accessories."
    },
    {
        "category_l1": "Electronics",
        "category_l2": "Computers",
        "full_category": "Electronics > Computers > Laptops",
        "description": "Laptops, notebook computers, Lenovo ThinkPad, Apple MacBook Pro, Dell XPS laptop computers."
    },
    {
        "category_l1": "Apparel & Accessories",
        "category_l2": "Clothing",
        "full_category": "Apparel & Accessories > Clothing > Shirts & Tops",
        "description": "Clothing, apparel, t-shirts, shirts, casual wear, polo shirts, hoodies, pants, denim jeans."
    },
    {
        "category_l1": "Home & Kitchen",
        "category_l2": "Kitchen Appliances",
        "full_category": "Home & Kitchen > Kitchen Appliances > Coffee Makers & Blenders",
        "description": "Kitchen appliances, coffee makers, blenders, air fryers, microwaves, electric kettles."
    }
]

INDEX_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "taxonomy_index.faiss"
METADATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "taxonomy_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"


class TaxonomyClassifier:
    """
    Semantic Product Taxonomy Classifier using Sentence Transformers and FAISS.
    """

    def __init__(self, model_name: str = MODEL_NAME, index_path: Optional[str] = None) -> None:
        self.model = SentenceTransformer(model_name)
        self.records: List[Dict[str, Any]] = TAXONOMY_RECORDS
        self.index_path = Path(index_path) if index_path else INDEX_PATH
        self.metadata_path = self.index_path.parent / "taxonomy_metadata.json"
        self.index: Optional[faiss.IndexFlatIP] = None

        self._load_or_build_index()

    def _load_or_build_index(self) -> None:
        # The cached index is reused only while its records equal the records in this module;
        # otherwise an edited description would never take effect.
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    meta = json.load(f)
                if meta.get("records") == self.records:
                    self.index = faiss.read_index(str(self.index_path))
                    return
            except Exception:
                pass

        self.build_index()

    def build_index(self) -> None:
        """
        Embeds every taxonomy record and writes the FAISS index and metadata to disk.

        Embeddings are L2 normalized so that inner product search is equivalent to
        cosine similarity.
        """
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        texts_to_embed = [rec["description"] for rec in self.records]
        embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w") as f:
            json.dump({"records": self.records}, f, indent=2)

    def classify_product(
        self,
        product_title: str,
        brand: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Classifies a product into the taxonomy, abstaining when no category is close.

        Returns the category fields plus `confidence` (cosine similarity to the best
        category), `margin` (how far the best category leads the runner-up), and
        `abstained`. When the best similarity is below `min_similarity` the category
        fields are "Unknown" and `nearest_category` records what would have been chosen.
        """
        floor = TAXONOMY_MIN_SIMILARITY if min_similarity is None else float(min_similarity)

        if not product_title and not brand:
            return {
                "category_l1": "Unknown",
                "category_l2": "Unknown",
                "full_category": "Unknown",
                "confidence": 0.0,
                "margin": 0.0,
                "abstained": True,
                "nearest_category": None,
            }

        query = f"{brand} {product_title}".strip() if brand else product_title.strip()
        query_vector = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

        k = min(2, len(self.records))
        distances, indices = self.index.search(query_vector, k)

        best_idx = int(indices[0][0])
        confidence = float(distances[0][0])
        margin = float(distances[0][0] - distances[0][1]) if k > 1 else confidence
        matched_rec = self.records[best_idx]

        if confidence < floor:
            return {
                "category_l1": "Unknown",
                "category_l2": "Unknown",
                "full_category": "Unknown",
                "confidence": round(confidence, 4),
                "margin": round(margin, 4),
                "abstained": True,
                "nearest_category": matched_rec["full_category"],
            }

        return {
            "category_l1": matched_rec["category_l1"],
            "category_l2": matched_rec["category_l2"],
            "full_category": matched_rec["full_category"],
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "abstained": False,
            "nearest_category": matched_rec["full_category"],
        }


# Prototype categories mapped into the supervised model's Amazon label space, used when the
# supervised model abstains.
PROTOTYPE_TO_AMAZON: Dict[str, Tuple[str, str]] = {
    "Smartphones": ("Cell Phones & Accessories", "Cell Phones"),
    "Audio": ("Electronics", "Headphones, Earbuds & Accessories"),
    "Wearables": ("Electronics", "Wearable Technology"),
    "Footwear": ("Clothing, Shoes & Jewelry", "Shoes"),
    "Storage": ("Electronics", "Computers & Accessories"),
    "Accessories": ("Electronics", "Computers & Accessories"),
    "Computers": ("Electronics", "Computers & Accessories"),
    "Clothing": ("Clothing, Shoes & Jewelry", "Clothing"),
    "Kitchen Appliances": ("Home & Kitchen", "Kitchen Appliances"),
}

LABEL_SEPARATOR = " > "

METHOD_SUPERVISED = "supervised"
METHOD_FALLBACK = "prototype_fallback"
METHOD_PROTOTYPE = "prototype"


def _unknown(confidence: float = 0.0, margin: float = 0.0, nearest: Optional[str] = None) -> Dict[str, Any]:
    return {
        "category_l1": "Unknown",
        "category_l2": "Unknown",
        "full_category": "Unknown",
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "abstained": True,
        "nearest_category": nearest,
    }


class SupervisedTaxonomyClassifier:
    """
    The trained taxonomy model with a prototype fallback.

    `artifact` is the dict written by `taxonomy_train.train_taxonomy`. The prototype
    classifier is built lazily and its sentence encoder is shared for the supervised queries,
    so only one encoder is loaded.
    """

    def __init__(self, artifact: Dict[str, Any], prototype: Optional[TaxonomyClassifier] = None) -> None:
        self.model = artifact["model"]
        self.labels: List[str] = list(artifact["labels"])
        self.tau = float(artifact["tau"])
        self.encoder_name = artifact.get("encoder_name", MODEL_NAME)
        self._prototype = prototype

    @property
    def prototype(self) -> TaxonomyClassifier:
        if self._prototype is None:
            self._prototype = TaxonomyClassifier(model_name=self.encoder_name)
        return self._prototype

    def predict_proba(self, queries: List[str]) -> np.ndarray:
        """Class probabilities for each query, columns in `self.labels` order."""
        vectors = self.prototype.model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
        return self.model.predict_proba(np.asarray(vectors, dtype=np.float32))

    def classify_product(
        self,
        product_title: str,
        brand: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Classifies a product. `confidence` is the top class probability and `margin` its lead
        over the second. Below `tau` the prototype classifier answers (`method` records which
        path answered); an empty title abstains. `min_similarity` applies to the fallback.
        """
        if not product_title or not product_title.strip():
            result = _unknown()
            result.update({"method": METHOD_SUPERVISED, "model_abstained": True, "model_label": None})
            return result
        query = f"{brand} {product_title}".strip() if brand else product_title.strip()
        probs = self.predict_proba([query])[0]
        order = np.argsort(probs)[::-1]
        top = float(probs[order[0]])
        second = float(probs[order[1]]) if len(order) > 1 else 0.0
        label = self.labels[int(order[0])]
        if top >= self.tau:
            l1, l2 = label.split(LABEL_SEPARATOR, 1)
            return {
                "category_l1": l1,
                "category_l2": l2,
                "full_category": label,
                "confidence": round(top, 4),
                "margin": round(top - second, 4),
                "abstained": False,
                "nearest_category": label,
                "method": METHOD_SUPERVISED,
                "model_abstained": False,
                "model_label": label,
            }

        fallback = self.prototype.classify_product(product_title, brand, min_similarity=min_similarity)
        extra = {"method": METHOD_FALLBACK, "model_abstained": True, "model_label": label,
                 "model_confidence": round(top, 4)}
        if fallback.get("abstained"):
            result = _unknown(float(fallback.get("confidence", 0.0)), float(fallback.get("margin", 0.0)),
                              fallback.get("nearest_category"))
            result.update(extra)
            return result
        l1, l2 = PROTOTYPE_TO_AMAZON.get(fallback["category_l2"], (fallback["category_l1"], fallback["category_l2"]))
        result = dict(fallback)
        result.update({"category_l1": l1, "category_l2": l2, "full_category": f"{l1}{LABEL_SEPARATOR}{l2}"})
        result.update(extra)
        return result


def load_taxonomy_artifact(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Loads the supervised taxonomy artifact, or returns None when it does not exist."""
    import joblib

    path = Path(path) if path is not None else Path(TAXONOMY_ARTIFACT_PATH)
    if not path.exists():
        return None
    return joblib.load(path)


# Global singletons
_classifier_instance: Optional[TaxonomyClassifier] = None
_supervised_instance: Optional[SupervisedTaxonomyClassifier] = None
_supervised_checked = False


def _prototype_classifier() -> TaxonomyClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = TaxonomyClassifier()
    return _classifier_instance


def get_supervised_classifier() -> Optional[SupervisedTaxonomyClassifier]:
    """The shared supervised classifier, or None when no artifact is present (checked once)."""
    global _supervised_instance, _supervised_checked
    if not _supervised_checked:
        artifact = load_taxonomy_artifact()
        if artifact is not None:
            _supervised_instance = SupervisedTaxonomyClassifier(artifact, prototype=_prototype_classifier())
        _supervised_checked = True
    return _supervised_instance


def reset_classifiers() -> None:
    """Forgets the shared classifiers, so the next call re-reads the artifact (tests)."""
    global _classifier_instance, _supervised_instance, _supervised_checked
    _classifier_instance = None
    _supervised_instance = None
    _supervised_checked = False


def classify_product(
    product_title: str,
    brand: Optional[str] = None,
    min_similarity: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Classifies a product with the supervised model when its artifact exists, otherwise with
    the prototype classifier. Both are built once per process and reused.
    """
    supervised = get_supervised_classifier()
    if supervised is not None:
        return supervised.classify_product(product_title, brand, min_similarity=min_similarity)
    result = _prototype_classifier().classify_product(product_title, brand, min_similarity=min_similarity)
    result["method"] = METHOD_PROTOTYPE
    return result

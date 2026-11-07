"""
Semantic product taxonomy classification with sentence-transformer embeddings and FAISS.

The label set is closed, so nearest-neighbour search alone always returns some category
however unrelated the product is. A similarity floor gives the classifier an abstain
path: below it the product is labelled Unknown, turning a silent misclassification into
a visible gap.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import TAXONOMY_MIN_SIMILARITY

# Mapping from representative taxonomy descriptions to structured L1/L2 and full hierarchy
# Aligned with Google Product Taxonomy standards and e-commerce categories
TAXONOMY_RECORDS = [
    {
        "category_l1": "Electronics",
        "category_l2": "Smartphones",
        "full_category": "Electronics > Communications > Telephony > Mobile Phones > Smartphones",
        "description": "Smartphones, mobile phone devices, Apple iPhone, Samsung Galaxy, Google Pixel, OnePlus Android mobile phones, 5G cellular handsets."
    },
    {
        "category_l1": "Electronics",
        "category_l2": "Audio",
        "full_category": "Electronics > Audio > Headphones & Earphones",
        "description": "Audio headphones, true wireless earbuds, boAt Rockerz, Airdopes, Sony wireless noise cancelling headphones, Apple AirPods, Bluetooth earphones."
    },
    {
        "category_l1": "Electronics",
        "category_l2": "Wearables",
        "full_category": "Electronics > Wearables > Smartwatches",
        "description": "Smartwatches and fitness trackers, Apple Watch, Samsung Galaxy Watch, Noise ColorFit smartwatch, health tracking wristbands."
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
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, "r") as f:
                    meta = json.load(f)
                    self.records = meta.get("records", TAXONOMY_RECORDS)
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


# Global singleton
_classifier_instance: Optional[TaxonomyClassifier] = None


def classify_product(
    product_title: str,
    brand: Optional[str] = None,
    min_similarity: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Classifies a product using a lazily constructed shared classifier.

    The classifier loads a sentence-transformer model and a FAISS index, so it is built
    once per process and reused rather than reconstructed on every call.
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = TaxonomyClassifier()
    return _classifier_instance.classify_product(product_title, brand, min_similarity=min_similarity)

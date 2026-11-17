"""Tests for process_batch: it must equal process() field by field, with the specified edge cases."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import List

import pytest
from hypothesis import given, settings, strategies as st

from src.ingestion import ner_extractor, taxonomy_classifier
from src.ingestion.catalog import build_catalog
from src.ingestion.entity_matcher import ProductRegistry
from src.ingestion.pipeline import ExtractedProduct, process, process_batch
from src.ingestion.screen_generator import render_listing

APPS = ["amazon_in", "flipkart", "amazon_us", "ebay"]
ACTIONS = ["view", "cart", "buy"]


def _build_texts(n: int = 200) -> tuple:
    from src.ingestion.synthetic_data import ACTION_SNIPPETS

    rng = random.Random(20260101)
    catalog = build_catalog()
    apps = [a for a in APPS if a in ACTION_SNIPPETS] or list(ACTION_SNIPPETS)
    texts: List[str] = []
    sources: List[str] = []
    for _ in range(n):
        item = catalog[rng.randrange(len(catalog))]
        app = apps[rng.randrange(len(apps))]
        action = [a for a in ACTIONS if a in ACTION_SNIPPETS[app]][rng.randrange(3)]
        lines = render_listing(item, app, action, rng.choice([0, 10, 25]), round(rng.uniform(3, 5), 1), rng)
        texts.append("\n".join(lines))
        sources.append(app)
    return texts, sources


_TEXTS, _SOURCES = _build_texts()
_FIXED_TS = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def _dump(p: ExtractedProduct, ignore_ts: bool) -> dict:
    d = p.model_dump()
    if ignore_ts:
        d.pop("timestamp", None)
    return d


def test_process_batch_equals_process_on_200_listings() -> None:
    assert len(_TEXTS) == 200
    batch = process_batch(_TEXTS, _SOURCES)
    single = [process(t, s) for t, s in zip(_TEXTS, _SOURCES)]
    assert len(batch) == 200
    for i, (b, s) in enumerate(zip(batch, single)):
        assert isinstance(b, ExtractedProduct)
        assert _dump(b, True) == _dump(s, True), f"row {i} differs"


def test_process_batch_equals_process_with_timestamps() -> None:
    stamps = [_FIXED_TS if i % 2 else None for i in range(50)]
    batch = process_batch(_TEXTS[:50], _SOURCES[:50], timestamps=stamps)
    for i, b in enumerate(batch):
        s = process(_TEXTS[i], _SOURCES[i], stamps[i])
        if stamps[i] is None:
            assert _dump(b, True) == _dump(s, True)
            assert b.timestamp is not None
        else:
            assert b.timestamp == _FIXED_TS
            assert _dump(b, False) == _dump(s, False)


def test_process_batch_default_app_source_is_unknown() -> None:
    batch = process_batch(_TEXTS[:5])
    for b, t in zip(batch, _TEXTS[:5]):
        assert b.app_source == "unknown"
        assert _dump(b, True) == _dump(process(t), True)


def test_process_batch_empty_returns_empty_list() -> None:
    assert process_batch([]) == []
    assert process_batch([], [], []) == []


def test_process_batch_single_row() -> None:
    out = process_batch(_TEXTS[:1], _SOURCES[:1])
    assert len(out) == 1
    assert _dump(out[0], True) == _dump(process(_TEXTS[0], _SOURCES[0]), True)


def test_process_batch_mismatched_app_sources_raises() -> None:
    with pytest.raises(ValueError):
        process_batch(_TEXTS[:3], _SOURCES[:2])
    with pytest.raises(ValueError):
        process_batch(_TEXTS[:3], _SOURCES[:4])


def test_process_batch_mismatched_timestamps_raises() -> None:
    with pytest.raises(ValueError):
        process_batch(_TEXTS[:3], _SOURCES[:3], timestamps=[None, None])
    with pytest.raises(ValueError):
        process_batch(_TEXTS[:2], timestamps=[None, None, None])


def test_process_batch_empty_texts_with_nonempty_args_raises() -> None:
    with pytest.raises(ValueError):
        process_batch([], ["amazon_in"])


@pytest.mark.parametrize("batch_size", [1, 3, 7, 256])
def test_process_batch_independent_of_batch_size(batch_size: int) -> None:
    texts, sources = _TEXTS[:20], _SOURCES[:20]
    got = process_batch(texts, sources, batch_size=batch_size)
    want = [process(t, s) for t, s in zip(texts, sources)]
    assert [_dump(g, True) for g in got] == [_dump(w, True) for w in want]


def test_process_batch_preserves_order_and_duplicates() -> None:
    texts = [_TEXTS[3], _TEXTS[1], _TEXTS[3], _TEXTS[2]]
    got = process_batch(texts)
    assert [g.raw_text for g in got] == texts
    assert _dump(got[0], True) == _dump(got[2], True)


def test_process_batch_with_registry_matches_sequential_process() -> None:
    texts, sources = _TEXTS[:25], _SOURCES[:25]
    reg_a, reg_b = ProductRegistry(), ProductRegistry()
    got = process_batch(texts, sources, registry=reg_a)
    want = [process(t, s, registry=reg_b) for t, s in zip(texts, sources)]
    assert [_dump(g, True) for g in got] == [_dump(w, True) for w in want]


def test_process_batch_handles_blank_and_noise_text() -> None:
    texts = ["", "   ", "zzz qqq", _TEXTS[0]]
    got = process_batch(texts)
    want = [process(t) for t in texts]
    assert [_dump(g, True) for g in got] == [_dump(w, True) for w in want]


@settings(max_examples=20, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=199), max_size=8), st.integers(min_value=1, max_value=5))
def test_process_batch_property_equals_process(idx: List[int], batch_size: int) -> None:
    texts = [_TEXTS[i] for i in idx]
    sources = [_SOURCES[i] for i in idx]
    got = process_batch(texts, sources, batch_size=batch_size)
    assert len(got) == len(texts)
    assert [_dump(g, True) for g in got] == [_dump(process(t, s), True) for t, s in zip(texts, sources)]


def test_spacy_pipeline_excludes_unused_components() -> None:
    nlp = ner_extractor._get_nlp() if hasattr(ner_extractor, "_get_nlp") else None
    if nlp is None:
        nlp = next(v for k, v in vars(ner_extractor).items() if hasattr(v, "pipe_names"))
    names = set(nlp.pipe_names)
    assert "ner" in names
    for gone in ("tagger", "parser", "lemmatizer", "attribute_ruler"):
        assert gone not in names


def test_taxonomy_query_cache_is_bounded_at_50000() -> None:
    funcs = [v for v in vars(taxonomy_classifier).values() if hasattr(v, "cache_info")]
    assert funcs, "expected an lru_cache on the taxonomy query encoder"
    assert any(f.cache_info().maxsize == 50_000 for f in funcs)


def test_encode_queries_matches_direct_encoding_row_by_row() -> None:
    import numpy as np

    taxonomy_classifier.reset_classifiers()
    clf = taxonomy_classifier.TaxonomyClassifier()
    taxonomy_classifier._ENCODERS[clf.model_name] = clf.model
    queries = ["Samsung Galaxy S24", "Sony WH-1000XM5 headphones", "Nike running shoes", "Samsung Galaxy S24"]
    got = taxonomy_classifier.encode_queries(clf.model_name, queries)
    want = clf.model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    assert got.shape == want.shape
    assert np.allclose(got, want, atol=1e-5)
    again = taxonomy_classifier.encode_queries(clf.model_name, queries[::-1])
    assert np.allclose(again, want[::-1], atol=1e-5)
    taxonomy_classifier.reset_classifiers()

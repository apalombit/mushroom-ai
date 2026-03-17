"""Integration tests for normalize_features semantic matching.

Requires the sentence-transformers model to be downloaded (no Docker needed).
These tests are excluded from CI unit test runs.
"""

import pytest

import ingestion.normalize as _norm
from ingestion.normalize import normalize_features


@pytest.fixture(autouse=True)
def enable_semantic_matching():
    original = _norm._SEMANTIC_MATCHING
    _norm._SEMANTIC_MATCHING = True
    yield
    _norm._SEMANTIC_MATCHING = original


def test_subfusiform_semantic_match():
    """'subfusiform' is not in aliases but should match 'fusiform' semantically."""
    # Note: subfusiform IS in aliases now, so this also validates alias resolution.
    # If the alias is removed, semantic should still catch it.
    features = {"spore": {"shape": "subfusiform"}}
    out = normalize_features(features)
    assert out["spore"]["shape"] == "fusiform"


def test_below_threshold_term_dropped():
    """A nonsense term should fall below threshold and be set to None."""
    features = {"cap": {"shape": "xyzzy_nonsense_blob"}}
    out = normalize_features(features)
    assert out["cap"]["shape"] is None

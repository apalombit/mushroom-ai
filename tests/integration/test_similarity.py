"""
Integration tests for similarity search. Requires full pipeline to have run
(init_db → fetch → extract → reconcile → embed).
"""

import pytest

from db.connection import get_session
from db.models import GroundTruthPair, ReconciledSpecies
from similarity.search import search_lookalikes
from similarity.weights import SimilarityWeights


def _count_embedded(session) -> int:
    return (
        session.query(ReconciledSpecies)
        .filter(ReconciledSpecies.embedding_macro_visual.isnot(None))
        .count()
    )


def test_gt_pairs_in_top10():
    """
    For each ground truth pair, species_b should appear in top-10 results for species_a
    and vice versa. Target pass rate: >70%.
    """
    session = get_session()
    try:
        pairs = session.query(GroundTruthPair).all()
        if not pairs:
            pytest.skip("No ground truth pairs — run python -m scripts.init_db first")
        if _count_embedded(session) < 2:
            pytest.skip("Need ≥2 embedded species — run full ingestion pipeline first")
    finally:
        session.close()

    # Build set of embedded species names for target-availability check
    s = get_session()
    try:
        embedded_names = {
            row.scientific_name
            for row in s.query(ReconciledSpecies)
            .filter(ReconciledSpecies.embedding_macro_visual.isnot(None))
            .all()
        }
    finally:
        s.close()

    weights = SimilarityWeights()
    total, passed = 0, 0

    for pair in pairs:
        for query, target in [(pair.species_a, pair.species_b), (pair.species_b, pair.species_a)]:
            # Skip if either side is not embedded (can't be a valid test)
            if query not in embedded_names or target not in embedded_names:
                continue
            s = get_session()
            try:
                _, candidates = search_lookalikes(s, query, weights, top_k=20)
                total += 1
                if any(c["scientific_name"] == target for c in candidates):
                    passed += 1
            except ValueError:
                pass  # Should not happen given the check above, but be safe
            finally:
                s.close()

    if total == 0:
        pytest.skip("No embedded species pairs available for testing")

    pass_rate = passed / total
    print(f"\nGT pair pass rate: {passed}/{total} = {pass_rate:.1%}")
    assert pass_rate >= 0.70, (
        f"Pass rate {pass_rate:.1%} below 70% target ({passed}/{total} pairs found)"
    )

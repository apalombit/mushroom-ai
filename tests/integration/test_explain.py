"""
Integration tests for LLM explanation generator.
Requires full pipeline to have run (init_db → fetch → extract → reconcile → embed).
Requires the configured LLM provider to be running.
"""

import pytest

from db.connection import get_session
from db.models import ReconciledSpecies
from llm.schemas import LookalikeExplanation
from similarity.explain import generate_explanation
from similarity.search import build_comparison_table, search_lookalikes
from similarity.weights import SimilarityWeights


def test_explanation_for_amanita_caesarea():
    """Full pipeline: search + explain for Amanita caesarea."""
    session = get_session()
    try:
        species = (
            session.query(ReconciledSpecies)
            .filter_by(scientific_name="Amanita caesarea")
            .first()
        )
        if species is None:
            pytest.skip("Amanita caesarea not in DB — run --reconcile first")
        if species.embedding_macro_visual is None:
            pytest.skip("Amanita caesarea not embedded — run --embed first")
    finally:
        session.close()

    s = get_session()
    try:
        weights = SimilarityWeights()
        query_species, candidates = search_lookalikes(s, "Amanita caesarea", weights, top_k=5)
        comparison_table = build_comparison_table(query_species, candidates)
    finally:
        s.close()

    explanation = generate_explanation("Amanita caesarea", comparison_table)

    assert isinstance(explanation, LookalikeExplanation)
    assert len(explanation.summary) > 0, "Summary should be non-empty"
    assert len(explanation.safety_warning) > 0, "Safety warning should be non-empty"
    assert isinstance(explanation.notable_pairs, list)

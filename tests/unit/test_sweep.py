"""Unit tests for evaluation/sweep.py — rescore logic and sweep helpers."""

from unittest.mock import MagicMock, patch

import numpy as np

from evaluation.sweep import (
    CandidateScores,
    QueryCache,
    precompute_scores,
    rescore,
)
from similarity.weights import WEIGHT_FIELDS


def _make_candidate(
    name: str,
    sims: dict[str, float],
    body_form_match: bool = True,
    hymenium_match: bool = True,
    size_class_match: bool = True,
    jaccard_sim: float = 0.0,
):
    """Helper: build a CandidateScores with defaults for missing WEIGHT_FIELDS."""
    full_sims = {f: 0.0 for f in WEIGHT_FIELDS}
    full_sims.update(sims)
    return CandidateScores(
        scientific_name=name,
        body_form_match=body_form_match,
        hymenium_match=hymenium_match,
        size_class_match=size_class_match,
        sims=full_sims,
        jaccard_sim=jaccard_sim,
    )


def _uniform_weights() -> dict[str, float]:
    """Equal weights across all 7 fields, summing to 1."""
    w = 1.0 / len(WEIGHT_FIELDS)
    return {f: w for f in WEIGHT_FIELDS}


# ---------------------------------------------------------------------------
# rescore tests
# ---------------------------------------------------------------------------


class TestRescore:
    def test_hit_at_rank_1(self):
        """Target is the highest-scored candidate -> recall 1.0 at all ks."""
        cache = [
            QueryCache(
                query_name="A",
                target_name="B",
                candidates=[
                    _make_candidate("B", {"macro_visual": 0.9, "structural": 0.9}),
                    _make_candidate("C", {"macro_visual": 0.1, "structural": 0.1}),
                ],
            )
        ]
        result = rescore(cache, _uniform_weights(), body_form_filter=False)
        assert result.recall_at[1] == 1.0
        assert result.hits_at[1] == 1

    def test_miss(self):
        """Target absent from candidates -> recall 0.0."""
        cache = [
            QueryCache(
                query_name="A",
                target_name="B",
                candidates=[
                    _make_candidate("C", {"macro_visual": 0.9}),
                    _make_candidate("D", {"macro_visual": 0.8}),
                ],
            )
        ]
        result = rescore(cache, _uniform_weights(), body_form_filter=False)
        assert result.recall_at[1] == 0.0
        assert result.recall_at[5] == 0.0
        assert result.hits_at[1] == 0

    def test_body_form_filter_excludes_target(self):
        """Target excluded when body_form_filter=True and body_form_match=False."""
        cache = [
            QueryCache(
                query_name="A",
                target_name="B",
                candidates=[
                    _make_candidate("B", {"macro_visual": 0.9}, body_form_match=False),
                    _make_candidate("C", {"macro_visual": 0.5}, body_form_match=True),
                ],
            )
        ]
        # With filter: B is excluded
        result_on = rescore(cache, _uniform_weights(), body_form_filter=True)
        assert result_on.recall_at[1] == 0.0

        # Without filter: B is included and ranks first
        result_off = rescore(cache, _uniform_weights(), body_form_filter=False)
        assert result_off.recall_at[1] == 1.0

    def test_weight_changes_ranking(self):
        """Different weights flip which candidate ranks first."""
        cache = [
            QueryCache(
                query_name="A",
                target_name="B",
                candidates=[
                    # B: high ecological, low cap_viz
                    _make_candidate("B", {"ecological": 0.9, "cap_viz": 0.1}),
                    # C: low ecological, high cap_viz
                    _make_candidate("C", {"ecological": 0.1, "cap_viz": 0.9}),
                ],
            )
        ]
        # Weights favoring cap_viz -> C ranks first, miss
        w_visual = {f: 0.0 for f in WEIGHT_FIELDS}
        w_visual["cap_viz"] = 1.0
        result_visual = rescore(cache, w_visual, body_form_filter=False)
        assert result_visual.recall_at[1] == 0.0

        # Weights favoring ecological -> B ranks first, hit
        w_eco = {f: 0.0 for f in WEIGHT_FIELDS}
        w_eco["ecological"] = 1.0
        result_eco = rescore(cache, w_eco, body_form_filter=False)
        assert result_eco.recall_at[1] == 1.0

    def test_error_entry_counts_toward_total(self):
        """Error entries count toward total but not hits."""
        cache = [
            QueryCache(
                query_name="A",
                target_name="B",
                candidates=[],
                error="Species not found",
            ),
            QueryCache(
                query_name="C",
                target_name="D",
                candidates=[
                    _make_candidate("D", {"macro_visual": 0.9}),
                ],
            ),
        ]
        result = rescore(cache, _uniform_weights(), body_form_filter=False)
        assert result.total == 2
        assert result.hits_at[1] == 1
        assert result.recall_at[1] == 0.5

    def test_alpha_one_ignores_jaccard(self):
        """alpha=1.0 means pure embedding — jaccard_sim should not affect ranking."""
        cache = [
            QueryCache(
                query_name="A",
                target_name="B",
                candidates=[
                    _make_candidate("B", {"ecological": 0.5}, jaccard_sim=1.0),
                    _make_candidate("C", {"ecological": 0.9}, jaccard_sim=0.0),
                ],
            )
        ]
        w = {f: 0.0 for f in WEIGHT_FIELDS}
        w["ecological"] = 1.0
        result = rescore(cache, w, body_form_filter=False, alpha=1.0)
        # C has higher embedding score, so B (target) is NOT rank 1
        assert result.recall_at[1] == 0.0

    def test_alpha_blends_jaccard(self):
        """alpha < 1.0 blends embedding and Jaccard scores."""
        cache = [
            QueryCache(
                query_name="A",
                target_name="B",
                candidates=[
                    # B: low embedding, high Jaccard
                    _make_candidate("B", {"ecological": 0.2}, jaccard_sim=0.9),
                    # C: high embedding, low Jaccard
                    _make_candidate("C", {"ecological": 0.8}, jaccard_sim=0.1),
                ],
            )
        ]
        w = {f: 0.0 for f in WEIGHT_FIELDS}
        w["ecological"] = 1.0
        # alpha=0.3 → 0.3*0.2 + 0.7*0.9 = 0.69 for B vs 0.3*0.8 + 0.7*0.1 = 0.31 for C
        result = rescore(cache, w, body_form_filter=False, alpha=0.3)
        assert result.recall_at[1] == 1.0

    def test_alpha_stored_in_weights(self):
        """Alpha value is recorded in the SweepResult weights dict."""
        cache = [
            QueryCache(query_name="A", target_name="B", candidates=[])
        ]
        result = rescore(cache, _uniform_weights(), body_form_filter=False, alpha=0.4)
        assert result.weights["alpha"] == 0.4


# ---------------------------------------------------------------------------
# Sweep helper tests
# ---------------------------------------------------------------------------


class TestSweepHelpers:
    def test_sweep_seed_reproducible(self):
        """Same seed produces identical results."""
        rng1 = np.random.default_rng(42)
        samples1 = rng1.dirichlet(np.ones(len(WEIGHT_FIELDS)), 10)

        rng2 = np.random.default_rng(42)
        samples2 = rng2.dirichlet(np.ones(len(WEIGHT_FIELDS)), 10)

        np.testing.assert_array_equal(samples1, samples2)

    def test_sample_configs_sum_to_one(self):
        """Dirichlet samples normalize correctly — each row sums to 1.0."""
        rng = np.random.default_rng(123)
        samples = rng.dirichlet(np.ones(len(WEIGHT_FIELDS)), 100)

        for row in samples:
            assert abs(sum(row) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# precompute_scores tests (mocked DB)
# ---------------------------------------------------------------------------


class TestPrecompute:
    def _mock_session(self, species_map: dict[str, MagicMock | None]):
        """Build a mock session that resolves species by name via ilike."""
        session = MagicMock()

        # Track last filter arg so .first() can resolve it
        _last_filter_name = [None]

        def _filter_side_effect(*args, **kwargs):
            # The filter arg is ReconciledSpecies.scientific_name.ilike(name)
            # which is a BinaryExpression mock — extract the right operand
            if args:
                expr = args[0]
                # Try to extract the string from the ilike clause
                right = getattr(expr, "right", None)
                if right is not None:
                    val = getattr(right, "value", None)
                    if val is not None:
                        _last_filter_name[0] = val

            mock_filtered = MagicMock()

            def _first():
                name = _last_filter_name[0]
                if name is None:
                    return None
                for sname, obj in species_map.items():
                    if sname.lower() == name.lower():
                        return obj
                return None

            mock_filtered.first = _first
            return mock_filtered

        mock_query = MagicMock()
        mock_query.filter = _filter_side_effect
        session.query = MagicMock(return_value=mock_query)

        return session

    def test_precompute_species_not_found(self):
        """Unknown species -> error recorded in QueryCache."""
        session = self._mock_session({})

        pair = MagicMock()
        pair.species_a = "Unknown"
        pair.species_b = "AlsoUnknown"

        cache = precompute_scores(session, [pair], pool_k=10)

        assert len(cache) == 2
        assert cache[0].error is not None
        assert "not found" in cache[0].error.lower()
        assert cache[0].candidates == []

    @patch("evaluation.sweep.search_by_group")
    @patch("evaluation.sweep.compute_numeric_similarity", return_value=0.5)
    def test_precompute_populates_sims(self, mock_numeric, mock_search):
        """Verify 7 sim keys per candidate after precompute."""
        # Create mock species
        query_sp = MagicMock()
        query_sp.id = 1
        query_sp.embedding_macro_visual = [0.1] * 384
        query_sp.features_json = {"overall_body_form": "agaricoid"}

        cand_sp = MagicMock()
        cand_sp.id = 2
        cand_sp.features_json = {"overall_body_form": "agaricoid"}

        session = self._mock_session({"QuerySp": query_sp, "TargetSp": cand_sp})
        session.get = MagicMock(return_value=cand_sp)

        # search_by_group returns candidate 2 for every group
        mock_search.return_value = [(2, "TargetSp", 0.8)]

        pair = MagicMock()
        pair.species_a = "QuerySp"
        pair.species_b = "TargetSp"

        cache = precompute_scores(session, [pair], pool_k=10)

        # First direction: QuerySp -> TargetSp
        entry = cache[0]
        assert entry.error is None
        assert len(entry.candidates) >= 1

        cand = entry.candidates[0]
        assert len(cand.sims) == len(WEIGHT_FIELDS)
        for f in WEIGHT_FIELDS:
            assert f in cand.sims

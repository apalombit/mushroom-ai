"""
Unit tests for similarity/aggregation.py.
No DB or LLM access required.
"""

import pytest

from ingestion.rubric import EMBEDDING_GROUPS
from similarity.aggregation import (
    aggregate_blended,
    aggregate_contrastive_gate,
    aggregate_rrf,
    aggregate_weighted_avg,
)
from similarity.weights import WEIGHT_FIELDS, SimilarityWeights

# Build a uniform weights object (all groups equal, numeric = 0.1)
_GROUPS = list(EMBEDDING_GROUPS.keys())
_N = len(_GROUPS)


def _make_weights(numeric: float = 0.1) -> SimilarityWeights:
    per_group = round((1.0 - numeric) / _N, 6)
    return SimilarityWeights(weights={g: per_group for g in _GROUPS}, numeric=numeric)


# ---------------------------------------------------------------------------
# weighted_avg
# ---------------------------------------------------------------------------


def test_weighted_avg_matches_manual():
    w = _make_weights()
    sims = {g: 0.5 for g in _GROUPS}
    sims["numeric"] = 0.4
    expected = sum(getattr(w, f) * sims.get(f, 0.0) for f in WEIGHT_FIELDS)
    result = aggregate_weighted_avg(sims, 0.4, w)
    assert abs(result - expected) < 1e-9


def test_weighted_avg_missing_group_uses_zero():
    w = _make_weights()
    # Only provide half the groups
    partial_sims = {g: 0.8 for g in _GROUPS[: _N // 2]}
    partial_sims["numeric"] = 0.3
    result = aggregate_weighted_avg(partial_sims, 0.3, w)
    assert 0.0 <= result <= 1.0


def test_weighted_avg_in_range():
    w = _make_weights()
    sims = {g: 1.0 for g in _GROUPS}
    sims["numeric"] = 1.0
    assert abs(aggregate_weighted_avg(sims, 1.0, w) - 1.0) < 1e-9
    sims_zero = {g: 0.0 for g in _GROUPS}
    sims_zero["numeric"] = 0.0
    assert aggregate_weighted_avg(sims_zero, 0.0, w) == 0.0


# ---------------------------------------------------------------------------
# rrf
# ---------------------------------------------------------------------------

_POOL_K = 60
_K_PARAM = 60


def _uniform_rank_maps(rank: int) -> dict[str, dict[int, int]]:
    """All groups assign the same rank to candidate 1."""
    return {g: {1: rank} for g in _GROUPS}


def test_rrf_rank1_all_groups_gives_normalized_1():
    rank_maps = _uniform_rank_maps(rank=1)
    result = aggregate_rrf(rank_maps, 1, _N, _POOL_K, 0.0, 0.0, k=_K_PARAM)
    assert abs(result - 1.0) < 1e-9


def test_rrf_absent_candidate_gives_worst_case():
    rank_maps = {g: {} for g in _GROUPS}  # candidate 99 never ranked
    result = aggregate_rrf(rank_maps, 99, _N, _POOL_K, 0.0, 0.0, k=_K_PARAM)
    result_rank1 = aggregate_rrf(_uniform_rank_maps(1), 1, _N, _POOL_K, 0.0, 0.0, k=_K_PARAM)
    assert result < result_rank1


def test_rrf_in_range():
    rank_maps = _uniform_rank_maps(rank=5)
    result = aggregate_rrf(rank_maps, 1, _N, _POOL_K, 0.1, 0.5, k=_K_PARAM)
    assert 0.0 <= result <= 1.0


def test_rrf_numeric_factored_in():
    rank_maps = _uniform_rank_maps(rank=1)
    # With numeric weight 0.2 and numeric_sim 0.0, embed portion should be 0.8
    result = aggregate_rrf(rank_maps, 1, _N, _POOL_K, 0.2, 0.0, k=_K_PARAM)
    assert abs(result - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# contrastive_gate
# ---------------------------------------------------------------------------


def _uniform_stats(mean: float = 0.3, std: float = 0.1) -> dict[str, tuple[float, float]]:
    return {g: (mean, std) for g in _GROUPS}


def test_contrastive_gate_all_pass_matches_weighted_avg():
    """z_threshold = -100 → all groups pass → must equal weighted_avg."""
    w = _make_weights(numeric=0.1)
    sims = {g: 0.6 for g in _GROUPS}
    numeric_sim = 0.4
    stats = _uniform_stats(mean=0.3, std=0.1)  # z = (0.6 - 0.3) / 0.1 = 3.0, all pass

    gate_result = aggregate_contrastive_gate(sims, numeric_sim, stats, w, z_threshold=-100.0)

    avg_sims = {**sims, "numeric": numeric_sim}
    avg_result = aggregate_weighted_avg(avg_sims, numeric_sim, w)
    assert abs(gate_result - avg_result) < 1e-9


def test_contrastive_gate_high_threshold_fallback_not_zero():
    """z_threshold = 10 → nothing passes → fallback to all groups (not 0)."""
    w = _make_weights(numeric=0.1)
    sims = {g: 0.5 for g in _GROUPS}
    stats = _uniform_stats(mean=0.5, std=0.1)  # z = 0.0 for all, none > 10
    result = aggregate_contrastive_gate(sims, 0.5, stats, w, z_threshold=10.0)
    assert result > 0.0


def test_contrastive_gate_below_avg_group_excluded():
    """Only one group below threshold: its contribution should be absent."""
    if _N < 2:
        pytest.skip("Need at least 2 groups for this test")

    w = _make_weights(numeric=0.0)  # numeric = 0 to isolate embedding part
    # First group is well below average; rest are above
    low_group = _GROUPS[0]
    high_groups = _GROUPS[1:]

    sims = {g: 0.8 for g in high_groups}
    sims[low_group] = 0.1
    stats = {g: (0.5, 0.1) for g in _GROUPS}
    # z for high groups: (0.8 - 0.5) / 0.1 = 3.0 (passes z_threshold=0)
    # z for low group: (0.1 - 0.5) / 0.1 = -4.0 (fails z_threshold=0)

    result_gated = aggregate_contrastive_gate(sims, 0.0, stats, w, z_threshold=0.0)
    # Without gating the low group, score should be higher
    result_all = aggregate_contrastive_gate(sims, 0.0, stats, w, z_threshold=-100.0)
    assert result_gated > result_all


def test_contrastive_gate_in_range():
    w = _make_weights()
    sims = {g: 0.7 for g in _GROUPS}
    stats = _uniform_stats()
    result = aggregate_contrastive_gate(sims, 0.5, stats, w, z_threshold=0.0)
    assert 0.0 <= result <= 1.0


def test_contrastive_gate_numeric_always_factored():
    """Numeric sim always contributes regardless of gate."""
    w = _make_weights(numeric=0.5)  # high numeric weight
    sims = {g: 0.0 for g in _GROUPS}  # all embeddings are 0
    stats = _uniform_stats(mean=0.0, std=0.1)
    # High z_threshold so nothing passes → fallback → embed_score ≈ 0
    result = aggregate_contrastive_gate(sims, 1.0, stats, w, z_threshold=100.0)
    # numeric_sim = 1.0 with weight 0.5 should contribute ~0.5
    assert result >= 0.4


# ---------------------------------------------------------------------------
# blended (embedding + Jaccard)
# ---------------------------------------------------------------------------


def test_blended_pure_embedding():
    """alpha=1.0 → pure embedding score."""
    assert aggregate_blended(0.8, 0.3, alpha=1.0) == pytest.approx(0.8)


def test_blended_pure_jaccard():
    """alpha=0.0 → pure Jaccard score."""
    assert aggregate_blended(0.8, 0.3, alpha=0.0) == pytest.approx(0.3)


def test_blended_half():
    """alpha=0.5 → average of both."""
    assert aggregate_blended(0.8, 0.4, alpha=0.5) == pytest.approx(0.6)


def test_blended_in_range():
    result = aggregate_blended(0.9, 0.5, alpha=0.7)
    assert 0.0 <= result <= 1.0

"""
Aggregation strategies for per-group similarity scores.

Three strategies:
- weighted_avg  : weighted sum over all groups (current default)
- rrf           : Reciprocal Rank Fusion — rank-based, insensitive to score scale
- contrastive_gate : weighted avg but gates groups whose z-score is below threshold
"""

from typing import Literal

from similarity.weights import WEIGHT_FIELDS, SimilarityWeights

AggregationStrategy = Literal["weighted_avg", "rrf", "contrastive_gate", "learned_ranker"]


def aggregate_weighted_avg(
    group_sims: dict[str, float],
    numeric_sim: float,  # noqa: ARG001 — kept for uniform call signature
    weights: SimilarityWeights,
) -> float:
    """Weighted average over all embedding groups + numeric similarity."""
    return sum(getattr(weights, f) * group_sims.get(f, 0.0) for f in WEIGHT_FIELDS)


def aggregate_rrf(
    group_rank_maps: dict[str, dict[int, int]],
    candidate_id: int,
    n_groups: int,
    pool_k: int,
    weights_numeric: float,
    numeric_sim: float,
    k: int = 60,
) -> float:
    """
    Reciprocal Rank Fusion across all embedding groups.

    Absent candidates get rank = pool_k + 1 (worst possible).
    Result is blended with numeric similarity using the numeric weight.
    """
    rrf_raw = sum(
        1.0 / (k + group_rank_maps[g].get(candidate_id, pool_k + 1)) for g in group_rank_maps
    )
    rrf_max = n_groups * (1.0 / (k + 1))
    rrf_normalized = rrf_raw / rrf_max if rrf_max > 0 else 0.0
    return rrf_normalized * (1 - weights_numeric) + numeric_sim * weights_numeric


def aggregate_contrastive_gate(
    group_sims: dict[str, float],
    numeric_sim: float,
    group_stats: dict[str, tuple[float, float]],
    weights: SimilarityWeights,
    z_threshold: float = 0.0,
) -> float:
    """
    Weighted average restricted to groups whose z-score exceeds z_threshold.

    z_i = (sim_i - mean_i) / std_i — groups below the threshold are gated out.
    Fallback: if no groups pass, use all groups (avoids returning 0).
    Numeric weight is always included in the final blend.
    Output stays in [0, 1] cosine space; z-scores are used only for filtering.
    """
    groups = [g for g in group_sims if g != "numeric"]

    # Gate: keep only groups whose z-score is above the threshold
    passing = []
    for g in groups:
        mean, std = group_stats.get(g, (0.0, 1e-6))
        z = (group_sims.get(g, 0.0) - mean) / max(std, 1e-6)
        if z > z_threshold:
            passing.append(g)

    # Fallback: if nothing passes, use all groups
    active = passing if passing else groups

    # Renormalize embedding weights over active groups
    total_w = sum(getattr(weights, g) for g in active)
    if total_w <= 0:
        embed_score = 0.0
    else:
        embed_score = sum((getattr(weights, g) / total_w) * group_sims.get(g, 0.0) for g in active)

    # Blend with numeric (original numeric weight)
    return embed_score * (1 - weights.numeric) + numeric_sim * weights.numeric


def aggregate_blended(
    embedding_score: float,
    jaccard_score: float,
    alpha: float,
) -> float:
    """Blend embedding-based score with Jaccard score.

    alpha=1.0 → pure embedding, alpha=0.0 → pure Jaccard.
    """
    return alpha * embedding_score + (1 - alpha) * jaccard_score

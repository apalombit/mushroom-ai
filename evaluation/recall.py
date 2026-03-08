"""
Recall@K evaluation of the similarity engine against ground truth pairs.

Core logic: for each ground truth pair (A, B), query lookalikes of A and check
if B appears in top-K (and vice versa). Computes Recall@1, @3, @5.
"""

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from db.models import GroundTruthPair, ReconciledSpecies
from ingestion.rubric import EMBEDDING_GROUPS
from similarity.search import search_lookalikes
from similarity.weights import SimilarityWeights

_first_embed_col = f"embedding_{next(iter(EMBEDDING_GROUPS))}"

logger = logging.getLogger(__name__)

# Named weight configs for benchmark sweeps
BENCHMARK_CONFIGS: dict[str, dict] = {
    "sweep-best": {
        "global": 0.0120, "cap_shape": 0.0137, "cap_top": 0.0310, "cap_margin": 0.0242,
        "cap_viz": 0.0895, "hymenium": 0.0297, "gills_arrange": 0.1121,
        "gills_distrib": 0.0683, "gills_viz": 0.0189, "pores": 0.0185,
        "stem": 0.0481, "stem_surf": 0.0582, "stem_viz": 0.0265, "veil": 0.0005,
        "volva": 0.0382, "flesh_visual": 0.0589, "flesh_inner": 0.0053,
        "uniformity": 0.0545, "latex": 0.0155, "flesh_perceptive": 0.0126,
        "spore_vis": 0.0274, "spore": 0.0485, "microscopic": 0.0170,
        "chemical": 0.0193, "ecological": 0.0310, "taxonomic": 0.1090,
        "numeric": 0.0118, "body_form_filter": False,
    },
    "visual-heavy": {
        "cap_viz": 0.20, "gills_arrange": 0.15, "gills_distrib": 0.10,
        "stem": 0.10, "volva": 0.10, "veil": 0.05, "spore_vis": 0.05,
        "ecological": 0.05, "taxonomic": 0.05, "numeric": 0.05,
        "body_form_filter": True,
    },
    "uniform": {
        # No explicit group weights → SimilarityWeights distributes evenly
        "numeric": 0.05, "body_form_filter": False,
    },
}


@dataclass
class PairResult:
    """Result of one directional query (A → B)."""

    query: str
    target: str
    rank: int | None  # 1-based rank if found, None if miss
    sim_overall: float | None = None
    sim_numeric: float | None = None
    group_sims: dict = field(default_factory=dict)  # {group_name: score}
    error: str | None = None


@dataclass
class EvalResult:
    """Aggregate evaluation results."""

    pair_results: list[PairResult] = field(default_factory=list)
    top_k: int = 5
    weights: dict[str, float | bool] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def total(self) -> int:
        return len(self.pair_results)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.pair_results if r.error is not None)

    def hits_at(self, k: int) -> int:
        return sum(1 for r in self.pair_results if r.rank is not None and r.rank <= k)

    def recall_at(self, k: int) -> float:
        if self.total == 0:
            return 0.0
        return self.hits_at(k) / self.total


def load_ground_truth(session: Session) -> list[GroundTruthPair]:
    """Load all ground truth pairs from the database."""
    return session.query(GroundTruthPair).all()


def evaluate_recall(
    session: Session,
    pairs: list[GroundTruthPair],
    weights: SimilarityWeights | None = None,
    top_k: int = 5,
) -> EvalResult:
    """
    Run bidirectional recall evaluation on ground truth pairs.

    For each pair, queries A→B and B→A, recording rank and similarity scores.
    Returns an EvalResult with all per-pair results and aggregate metrics.
    """
    if weights is None:
        weights = SimilarityWeights()

    results: list[PairResult] = []
    start = time.monotonic()

    for pair in pairs:
        for query_name, target_name in [
            (pair.species_a, pair.species_b),
            (pair.species_b, pair.species_a),
        ]:
            result = _evaluate_single(session, query_name, target_name, weights, top_k)
            results.append(result)

    duration = time.monotonic() - start

    return EvalResult(
        pair_results=results,
        top_k=top_k,
        weights=weights.as_dict(),
        duration_s=round(duration, 1),
    )


def _evaluate_single(
    session: Session,
    query_name: str,
    target_name: str,
    weights: SimilarityWeights,
    top_k: int,
) -> PairResult:
    """Evaluate a single directional query."""
    try:
        _, candidates = search_lookalikes(session, query_name, weights, top_k=top_k)
    except ValueError as e:
        logger.warning("Error querying %s: %s", query_name, e)
        return PairResult(query=query_name, target=target_name, rank=None, error=str(e))

    names = [c["scientific_name"] for c in candidates]
    if target_name in names:
        rank = names.index(target_name) + 1
        cand = candidates[rank - 1]
        group_sims = {g: cand.get(f"similarity_{g}") for g in EMBEDDING_GROUPS}
        return PairResult(
            query=query_name,
            target=target_name,
            rank=rank,
            sim_overall=cand.get("similarity_overall"),
            sim_numeric=cand.get("similarity_numeric"),
            group_sims=group_sims,
        )

    return PairResult(query=query_name, target=target_name, rank=None)


def dataset_stats(session: Session) -> dict[str, int]:
    """Return counts for params logging: total species, embedded species."""
    total = session.query(ReconciledSpecies).count()
    embedded = (
        session.query(ReconciledSpecies)
        .filter(getattr(ReconciledSpecies, _first_embed_col).isnot(None))
        .count()
    )
    gt_pairs = session.query(GroundTruthPair).count()
    return {"species_count": total, "embedded_count": embedded, "gt_pairs_count": gt_pairs}

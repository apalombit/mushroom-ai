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
from similarity.search import search_lookalikes
from similarity.weights import SimilarityWeights

logger = logging.getLogger(__name__)

# Named weight configs for benchmark sweeps
BENCHMARK_CONFIGS: dict[str, dict] = {
    "default": {
        "macro_visual": 0.69,
        "structural": 0.12,
        "flesh_sensory": 0.07,
        "microscopic_lab": 0.02,
        "ecological": 0.01,
        "taxonomic": 0.01,
        "numeric": 0.10,
        "body_form_filter": False,
    },
    "visual-heavy": {
        "macro_visual": 0.50,
        "structural": 0.20,
        "flesh_sensory": 0.02,
        "microscopic_lab": 0.01,
        "ecological": 0.10,
        "taxonomic": 0.02,
        "numeric": 0.10,
        "body_form_filter": True,
    },
    "balanced": {
        "macro_visual": 0.15,
        "structural": 0.15,
        "flesh_sensory": 0.10,
        "microscopic_lab": 0.10,
        "ecological": 0.15,
        "taxonomic": 0.10,
        "numeric": 0.15,
        "body_form_filter": False,
    },
    "no-filter": {
        "macro_visual": 0.30,
        "structural": 0.15,
        "flesh_sensory": 0.05,
        "microscopic_lab": 0.03,
        "ecological": 0.15,
        "taxonomic": 0.05,
        "numeric": 0.12,
        "body_form_filter": False,
    },
}


@dataclass
class PairResult:
    """Result of one directional query (A → B)."""

    query: str
    target: str
    rank: int | None  # 1-based rank if found, None if miss
    sim_overall: float | None = None
    sim_macro_visual: float | None = None
    sim_structural: float | None = None
    sim_flesh_sensory: float | None = None
    sim_microscopic_lab: float | None = None
    sim_ecological: float | None = None
    sim_taxonomic: float | None = None
    sim_numeric: float | None = None
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
        return PairResult(
            query=query_name,
            target=target_name,
            rank=rank,
            sim_overall=cand.get("similarity_overall"),
            sim_macro_visual=cand.get("similarity_macro_visual"),
            sim_structural=cand.get("similarity_structural"),
            sim_flesh_sensory=cand.get("similarity_flesh_sensory"),
            sim_microscopic_lab=cand.get("similarity_microscopic_lab"),
            sim_ecological=cand.get("similarity_ecological"),
            sim_taxonomic=cand.get("similarity_taxonomic"),
            sim_numeric=cand.get("similarity_numeric"),
        )

    return PairResult(query=query_name, target=target_name, rank=None)


def dataset_stats(session: Session) -> dict[str, int]:
    """Return counts for params logging: total species, embedded species."""
    total = session.query(ReconciledSpecies).count()
    embedded = (
        session.query(ReconciledSpecies)
        .filter(ReconciledSpecies.embedding_macro_visual.isnot(None))
        .count()
    )
    gt_pairs = session.query(GroundTruthPair).count()
    return {"species_count": total, "embedded_count": embedded, "gt_pairs_count": gt_pairs}

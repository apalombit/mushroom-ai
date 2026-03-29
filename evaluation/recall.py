"""
Recall@K evaluation of the similarity engine against known lookalike pairs.

Core logic: for each known pair (A, B), query lookalikes of A and check
if B appears in top-K (and vice versa). Computes Recall@1, @3, @5.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from db.models import GroundTruthPair, ReconciledSpecies
from ingestion.rubric import EMBEDDING_GROUPS
from similarity.search import search_lookalikes
from similarity.weights import SimilarityWeights

_first_embed_col = f"embedding_{next(iter(EMBEDDING_GROUPS))}"
_KNOWN_LOOKALIKES_YAML = (
    Path(__file__).resolve().parent.parent / "data" / "seed" / "known_lookalikes.yaml"
)

logger = logging.getLogger(__name__)

# Named weight configs for benchmark sweeps
BENCHMARK_CONFIGS: dict[str, dict] = {
    # finer_grps sweep #2: R@1=9.7% R@3=16.1% R@5=24.2% (seed=42, n=10000, BFF=False)
    "finer-sweep-best": {
        "size": 0.0073,
        "shape": 0.0166,
        "cap_color": 0.0933,
        "stem_color": 0.0218,
        "cap_shape": 0.0101,
        "cap_depress": 0.0035,
        "cap_feel": 0.0036,
        "cap_text": 0.0076,
        "cap_ornaments": 0.0182,
        "cap_margin": 0.0014,
        "cap_viz": 0.0196,
        "cap_bruise": 0.0290,
        "hymenium": 0.0284,
        "gills_attach": 0.1382,
        "gills_distrib": 0.0301,
        "gills_viz": 0.0096,
        "pores": 0.0015,
        "pores_bruise": 0.0149,
        "stem_shape": 0.0030,
        "stem_attach": 0.0032,
        "stem": 0.0063,
        "stem_surf": 0.0040,
        "stem_bruise": 0.0258,
        "stem_age": 0.0068,
        "stem_basecol": 0.0099,
        "veil": 0.0257,
        "cortina": 0.0571,
        "veil_colour": 0.0255,
        "veil_shape": 0.0194,
        "ring_pos": 0.0027,
        "ring_pers": 0.0284,
        "ring_mov": 0.0109,
        "volva": 0.0219,
        "volva_shape": 0.0209,
        "volva_col": 0.0014,
        "flesh_visual": 0.0076,
        "flesh_inner": 0.0445,
        "uniformity": 0.0521,
        "latex": 0.0209,
        "flesh_perceptive": 0.0076,
        "spore_vis": 0.0040,
        "ecological": 0.0107,
        "habitat": 0.0037,
        "trees": 0.0121,
        "growth": 0.0191,
        "taxonomic": 0.0143,
        "numeric": 0.0759,
        "body_form_filter": False,
        "group_filter": False,
        "morpho_pool_required": False,
        "alpha": 1.0,
    },
    # Blended sweep: R@1=9.9% R@3=21.1% R@5=31.7% (seed=42, n=5000, 620 edges)
    "blended-best": {
        "size": 0.0085,
        "shape": 0.0042,
        "cap_color": 0.0661,
        "stem_color": 0.0134,
        "cap_shape": 0.0109,
        "cap_depress": 0.0046,
        "cap_feel": 0.0108,
        "cap_text": 0.0125,
        "cap_ornaments": 0.0072,
        "cap_margin": 0.0340,
        "cap_viz": 0.0026,
        "cap_bruise": 0.0086,
        "hymenium": 0.0166,
        "gills_attach": 0.0135,
        "gills_distrib": 0.0054,
        "gills_viz": 0.0119,
        "pores": 0.0902,
        "pores_bruise": 0.0165,
        "stem_shape": 0.0113,
        "stem_attach": 0.0072,
        "stem": 0.0158,
        "stem_surf": 0.0002,
        "stem_bruise": 0.0105,
        "stem_age": 0.0050,
        "stem_basecol": 0.0518,
        "veil": 0.0170,
        "cortina": 0.0377,
        "veil_colour": 0.0102,
        "veil_shape": 0.0297,
        "ring_pos": 0.0018,
        "ring_pers": 0.0650,
        "ring_mov": 0.0010,
        "volva": 0.0684,
        "volva_shape": 0.0111,
        "volva_col": 0.0275,
        "flesh_visual": 0.0135,
        "flesh_inner": 0.0014,
        "uniformity": 0.0278,
        "latex": 0.0062,
        "flesh_perceptive": 0.0671,
        "spore_vis": 0.0014,
        "ecological": 0.0081,
        "habitat": 0.0545,
        "trees": 0.0103,
        "growth": 0.0015,
        "taxonomic": 0.0802,
        "numeric": 0.0195,
        "body_form_filter": True,
        "group_filter": False,
        "morpho_pool_required": False,
        "alpha": 0.5,
    },
    "uniform": {
        # No explicit group weights → SimilarityWeights distributes evenly
        "numeric": 0.05,
        "body_form_filter": False,
        "group_filter": False,
        "morpho_pool_required": False,
        "alpha": 1.0,
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
    strategy: str = "weighted_avg"

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
    strategy: str = "weighted_avg",
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
            result = _evaluate_single(session, query_name, target_name, weights, top_k, strategy)
            results.append(result)

    duration = time.monotonic() - start

    return EvalResult(
        pair_results=results,
        top_k=top_k,
        weights=weights.as_dict(),
        duration_s=round(duration, 1),
        strategy=strategy,
    )


def _evaluate_single(
    session: Session,
    query_name: str,
    target_name: str,
    weights: SimilarityWeights,
    top_k: int,
    strategy: str = "weighted_avg",
) -> PairResult:
    """Evaluate a single directional query."""
    try:
        _, candidates = search_lookalikes(
            session, query_name, weights, top_k=top_k, strategy=strategy
        )
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


# ---------------------------------------------------------------------------
# Lookalike graph — known lookalikes from known_lookalikes.yaml
# ---------------------------------------------------------------------------


def load_lookalike_graph(
    yaml_path: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Load known_lookalikes.yaml as bidirectional (species, lookalike) edges.

    Only includes species from the 'indexed' list (present in species_list.yaml
    and thus in our DB). Returns deduplicated edges sorted alphabetically.
    """
    path = Path(yaml_path) if yaml_path else _KNOWN_LOOKALIKES_YAML
    with open(path) as f:
        data = yaml.safe_load(f)

    seen: set[tuple[str, str]] = set()
    edges: list[tuple[str, str]] = []
    for entry in data.get("lookalikes", []):
        species = entry["species"]
        for lookalike in entry.get("indexed", []):
            if species == lookalike:
                continue
            pair = tuple(sorted([species, lookalike]))
            if pair not in seen:
                seen.add(pair)
                edges.append(pair)

    edges.sort()
    return edges


def load_unified_lookalikes(session: Session) -> list[tuple[str, str]]:
    """Load known_lookalikes.yaml edges, filtered to species present in the DB.

    Resolves case mismatches against DB species names. Drops pairs where either
    species is missing from the DB. Returns deduplicated, sorted edges.
    """
    db_names = {
        r[0]
        for r in session.execute(
            __import__("sqlalchemy").text(
                "SELECT scientific_name FROM reconciled_species"
            )
        ).fetchall()
    }
    db_lower = {s.lower(): s for s in db_names}

    def _resolve(name: str) -> str | None:
        if name in db_names:
            return name
        return db_lower.get(name.lower())

    seen: set[tuple[str, str]] = set()

    with open(_KNOWN_LOOKALIKES_YAML) as f:
        data = yaml.safe_load(f)
    for entry in data.get("lookalikes", []):
        species = _resolve(entry["species"])
        if not species:
            continue
        for lk in entry.get("indexed", []):
            target = _resolve(lk)
            if target and target != species:
                seen.add(tuple(sorted([species, target])))

    edges = sorted(seen)
    logger.info("Unified lookalikes: %d edges (from known_lookalikes.yaml)", len(edges))
    return edges


def evaluate_recall_on_graph(
    session: Session,
    edges: list[tuple[str, str]],
    weights: SimilarityWeights | None = None,
    top_k: int = 5,
    strategy: str = "weighted_avg",
) -> EvalResult:
    """Run bidirectional recall evaluation on lookalike graph edges.

    Same logic as evaluate_recall() but takes (species_a, species_b) tuples
    instead of GroundTruthPair ORM objects. Also reports per-genus breakdown.
    """
    if weights is None:
        weights = SimilarityWeights()

    results: list[PairResult] = []
    start = time.monotonic()

    for species_a, species_b in edges:
        for query_name, target_name in [(species_a, species_b), (species_b, species_a)]:
            result = _evaluate_single(session, query_name, target_name, weights, top_k, strategy)
            results.append(result)

    duration = time.monotonic() - start

    return EvalResult(
        pair_results=results,
        top_k=top_k,
        weights=weights.as_dict(),
        duration_s=round(duration, 1),
        strategy=strategy,
    )


def per_genus_breakdown(eval_result: EvalResult, k: int = 5) -> dict[str, dict[str, float]]:
    """Break down recall by genus (first word of the query species name).

    Returns {genus: {"total": N, "hits": N, "recall": float}}.
    """
    genus_results: dict[str, list[PairResult]] = defaultdict(list)
    for r in eval_result.pair_results:
        genus = r.query.split()[0] if r.query else "unknown"
        genus_results[genus].append(r)

    breakdown: dict[str, dict[str, float]] = {}
    for genus, results in sorted(genus_results.items()):
        total = len(results)
        hits = sum(1 for r in results if r.rank is not None and r.rank <= k)
        breakdown[genus] = {
            "total": total,
            "hits": hits,
            "recall": hits / total if total > 0 else 0.0,
        }
    return breakdown

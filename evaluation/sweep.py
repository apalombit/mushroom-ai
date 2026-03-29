"""
Weight sweep optimizer: precompute per-group similarities once, then rescore
with thousands of weight configs as pure arithmetic.

Precompute phase: ~60-90 min for 620 pairs (pgvector + Jaccard).
Rescore phase: vectorized numpy — 200K configs in ~30s.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from db.models import GroundTruthPair, ReconciledSpecies
from ingestion.normalize import load_vocabulary
from ingestion.rubric import EMBEDDING_GROUPS, _get_nested
from similarity.jaccard import compute_soft_jaccard
from similarity.numeric import compute_numeric_similarity
from similarity.search import _GROUP_COLUMN, search_by_group
from similarity.weights import WEIGHT_FIELDS

_first_embed_col = f"embedding_{next(iter(EMBEDDING_GROUPS))}"
_N_FIELDS = len(WEIGHT_FIELDS)

logger = logging.getLogger(__name__)


@dataclass
class CandidateScores:
    scientific_name: str
    body_form_match: bool
    hymenium_match: bool
    size_class_match: bool
    sims: dict[str, float]  # keys matching WEIGHT_FIELDS
    jaccard_sim: float = 0.0


@dataclass
class QueryCache:
    query_name: str
    target_name: str
    candidates: list[CandidateScores]
    error: str | None = None


@dataclass
class VectorizedEntry:
    """Numpy-friendly representation of a QueryCache entry for fast rescoring."""
    target_idx: int  # index of target in names, or -1 if absent
    sim_matrix: np.ndarray  # shape (n_candidates, n_fields)
    jaccard_vec: np.ndarray  # shape (n_candidates,)
    body_form_mask: np.ndarray  # shape (n_candidates,) bool — True = passes filter
    hymenium_mask: np.ndarray
    size_class_mask: np.ndarray
    n_candidates: int = 0


@dataclass
class SweepResult:
    weights: dict[str, float | bool]
    recall_at: dict[int, float]  # {1: 0.17, 3: 0.30, 5: 0.37}
    hits_at: dict[int, int]
    total: int


def precompute_scores(
    session: Session,
    pairs: list[GroundTruthPair],
    pool_k: int = 60,
    compute_jaccard: bool = True,
) -> list[QueryCache]:
    """
    Precompute per-group similarity scores for all ground truth pairs (both directions).

    Returns a list of QueryCache entries with raw similarity scores per candidate,
    independent of weight configuration.
    """
    vocab = load_vocabulary() if compute_jaccard else None
    cache: list[QueryCache] = []

    # Deduplicate queries: same species may appear in multiple pairs
    all_directions: list[tuple[str, str]] = []
    for pair in pairs:
        all_directions.append((pair.species_a, pair.species_b))
        all_directions.append((pair.species_b, pair.species_a))

    # Cache species lookups to avoid repeated DB queries
    species_cache: dict[str, ReconciledSpecies | None] = {}

    def _get_species(name: str) -> ReconciledSpecies | None:
        if name not in species_cache:
            species_cache[name] = (
                session.query(ReconciledSpecies)
                .filter(ReconciledSpecies.scientific_name.ilike(name))
                .first()
            )
        return species_cache[name]

    # Cache per-species group search results: (species_id, group) -> {cand_id: (name, score)}
    group_cache: dict[tuple[int, str], dict[int, tuple[str, float]]] = {}

    for query_name, target_name in all_directions:
        query_species = _get_species(query_name)
        if query_species is None:
            cache.append(
                QueryCache(
                    query_name=query_name,
                    target_name=target_name,
                    candidates=[],
                    error=f"Species not found: {query_name!r}",
                )
            )
            continue

        if getattr(query_species, _first_embed_col) is None:
            cache.append(
                QueryCache(
                    query_name=query_name,
                    target_name=target_name,
                    candidates=[],
                    error=f"Species {query_name!r} has no embeddings",
                )
            )
            continue

        # Run 6 embedding searches (cached per species)
        group_maps: dict[str, dict[int, tuple[str, float]]] = {}
        for group in _GROUP_COLUMN:
            key = (query_species.id, group)
            if key not in group_cache:
                results = search_by_group(session, query_species.id, group, pool_k)
                group_cache[key] = {sid: (name, score) for sid, name, score in results}
            group_maps[group] = group_cache[key]

        # Collect all candidate IDs across groups
        all_ids: set[int] = set()
        for gmap in group_maps.values():
            all_ids.update(gmap)

        query_features = query_species.features_json or {}
        query_body_form = _get_nested(query_features, "overall_body_form")
        query_hymenium = _get_nested(query_features, "hymenium.type")
        query_size_class = _get_nested(query_features, "overall_size_class")

        candidates: list[CandidateScores] = []
        for sid in all_ids:
            # Get candidate name
            name = None
            for gmap in group_maps.values():
                if sid in gmap:
                    name = gmap[sid][0]
                    break

            species_row = session.get(ReconciledSpecies, sid)
            cand_features = (species_row.features_json if species_row else {}) or {}

            # Precompute gating matches
            body_form_match = True
            if query_body_form:
                cand_body_form = _get_nested(cand_features, "overall_body_form")
                if cand_body_form and cand_body_form.lower() != query_body_form.lower():
                    body_form_match = False

            hymenium_match = True
            if query_hymenium:
                cand_hymenium = _get_nested(cand_features, "hymenium.type")
                if cand_hymenium and cand_hymenium.lower() != query_hymenium.lower():
                    hymenium_match = False

            size_class_match = True
            if query_size_class:
                cand_size = _get_nested(cand_features, "overall_size_class")
                if cand_size and cand_size.lower() != query_size_class.lower():
                    size_class_match = False

            # Embedding similarities (6 groups)
            sims: dict[str, float] = {}
            for group in _GROUP_COLUMN:
                sims[group] = group_maps[group].get(sid, (None, 0.0))[1]

            # Numeric similarity
            sims["numeric"] = compute_numeric_similarity(query_features, cand_features)

            # Jaccard similarity
            jaccard_sim = 0.0
            if vocab is not None:
                jaccard_sim, _ = compute_soft_jaccard(
                    query_features, cand_features, vocab
                )

            candidates.append(
                CandidateScores(
                    scientific_name=name,
                    body_form_match=body_form_match,
                    hymenium_match=hymenium_match,
                    size_class_match=size_class_match,
                    sims=sims,
                    jaccard_sim=jaccard_sim,
                )
            )

        cache.append(
            QueryCache(
                query_name=query_name,
                target_name=target_name,
                candidates=candidates,
            )
        )

    return cache


def _vectorize_cache(cache: list[QueryCache]) -> list[VectorizedEntry | None]:
    """Convert QueryCache list to numpy arrays for fast vectorized rescoring."""
    vectorized: list[VectorizedEntry | None] = []
    for entry in cache:
        if entry.error or not entry.candidates:
            vectorized.append(None)
            continue

        n = len(entry.candidates)
        sim_matrix = np.zeros((n, _N_FIELDS), dtype=np.float64)
        jaccard_vec = np.zeros(n, dtype=np.float64)
        body_form_mask = np.ones(n, dtype=bool)
        hymenium_mask = np.ones(n, dtype=bool)
        size_class_mask = np.ones(n, dtype=bool)
        target_idx = -1

        for i, cand in enumerate(entry.candidates):
            for j, f in enumerate(WEIGHT_FIELDS):
                sim_matrix[i, j] = cand.sims.get(f, 0.0)
            jaccard_vec[i] = cand.jaccard_sim
            body_form_mask[i] = cand.body_form_match
            hymenium_mask[i] = cand.hymenium_match
            size_class_mask[i] = cand.size_class_match
            if cand.scientific_name == entry.target_name:
                target_idx = i

        vectorized.append(VectorizedEntry(
            target_idx=target_idx,
            sim_matrix=sim_matrix,
            jaccard_vec=jaccard_vec,
            body_form_mask=body_form_mask,
            hymenium_mask=hymenium_mask,
            size_class_mask=size_class_mask,
            n_candidates=n,
        ))

    return vectorized


def rescore(
    cache: list[QueryCache],
    weights: dict[str, float],
    body_form_filter: bool,
    hymenium_filter: bool = True,
    size_class_filter: bool = True,
    recall_ks: tuple[int, ...] = (1, 3, 5),
    top_k: int = 5,
    alpha: float = 1.0,
) -> SweepResult:
    """
    Pure arithmetic rescoring — no DB access. Called thousands of times.

    For each QueryCache entry: optionally filter by body_form/hymenium/size_class
    match, compute weighted overall score, sort, check if target in top-K.
    alpha blends embedding score (alpha) with Jaccard (1-alpha).
    """
    hits: dict[int, int] = {k: 0 for k in recall_ks}
    total = 0

    for entry in cache:
        total += 1

        if entry.error:
            continue

        # Filter and score candidates
        scored: list[tuple[str, float]] = []
        for cand in entry.candidates:
            if body_form_filter and not cand.body_form_match:
                continue
            if hymenium_filter and not cand.hymenium_match:
                continue
            if size_class_filter and not cand.size_class_match:
                continue
            overall = sum(weights.get(f, 0.0) * cand.sims.get(f, 0.0) for f in WEIGHT_FIELDS)
            if alpha < 1.0:
                overall = alpha * overall + (1 - alpha) * cand.jaccard_sim
            scored.append((cand.scientific_name, overall))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Check if target in top-K for each recall threshold
        names = [name for name, _ in scored]
        for k in recall_ks:
            if entry.target_name in names[:k]:
                hits[k] += 1

    recall = {k: hits[k] / total if total > 0 else 0.0 for k in recall_ks}

    weight_dict: dict[str, float | bool] = {f: weights.get(f, 0.0) for f in WEIGHT_FIELDS}
    weight_dict["body_form_filter"] = body_form_filter
    weight_dict["hymenium_filter"] = hymenium_filter
    weight_dict["size_class_filter"] = size_class_filter
    weight_dict["alpha"] = alpha

    return SweepResult(weights=weight_dict, recall_at=recall, hits_at=hits, total=total)


def _rescore_vectorized(
    v_cache: list[VectorizedEntry | None],
    weight_vec: np.ndarray,
    body_form_filter: bool,
    hymenium_filter: bool,
    size_class_filter: bool,
    recall_ks: tuple[int, ...],
    top_k: int,
    alpha: float,
) -> tuple[dict[int, int], int]:
    """Vectorized rescore using precomputed numpy arrays. Returns (hits, total)."""
    hits = {k: 0 for k in recall_ks}
    total = len(v_cache)

    for ve in v_cache:
        if ve is None:
            continue
        if ve.target_idx < 0:
            continue

        # Compute embedding scores: matrix @ weights
        scores = ve.sim_matrix @ weight_vec

        # Alpha blend with Jaccard
        if alpha < 1.0:
            scores = alpha * scores + (1.0 - alpha) * ve.jaccard_vec

        # Apply gate filters by setting excluded candidates to -inf
        if body_form_filter:
            scores[~ve.body_form_mask] = -np.inf
        if hymenium_filter:
            scores[~ve.hymenium_mask] = -np.inf
        if size_class_filter:
            scores[~ve.size_class_mask] = -np.inf

        # Check if target is in top-K by score
        target_score = scores[ve.target_idx]
        if np.isinf(target_score) and target_score < 0:
            continue  # target was filtered out

        for k in recall_ks:
            # Count how many candidates score strictly higher than target
            n_better = int(np.sum(scores > target_score))
            if n_better < k:
                hits[k] += 1

    return hits, total


def sweep_weights(
    session: Session,
    pairs: list[GroundTruthPair],
    n_configs: int = 200,
    recall_ks: tuple[int, ...] = (1, 3, 5),
    top_k: int = 5,
    pool_k: int = 60,
    seed: int = 42,
    alphas: list[float] | None = None,
) -> list[SweepResult]:
    """
    Main entry point. Precomputes scores once, then sweeps random weight configs.

    Generates n_configs weight vectors via Dirichlet distribution, tests each
    with filter combos and alpha values using vectorized numpy rescoring.
    Returns results sorted by recall@top_k descending.
    """
    if alphas is None:
        alphas = [1.0, 0.7, 0.5, 0.3, 0.2]

    logger.info("Precomputing scores for %d pairs...", len(pairs))
    cache = precompute_scores(session, pairs, pool_k=pool_k)

    logger.info("Vectorizing cache for fast rescoring...")
    v_cache = _vectorize_cache(cache)

    n_combos = n_configs * 8 * len(alphas)
    logger.info(
        "Precompute done. Sweeping %d configs (x8 filter combos x%d alphas = %d)...",
        n_configs, len(alphas), n_combos,
    )

    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(np.ones(len(WEIGHT_FIELDS)), n_configs)

    results: list[SweepResult] = []
    for idx, sample in enumerate(samples):
        weight_vec = sample.astype(np.float64)
        weights_dict = {f: float(sample[i]) for i, f in enumerate(WEIGHT_FIELDS)}
        for bff in (True, False):
            for hf in (True, False):
                for scf in (True, False):
                    for a in alphas:
                        hits, total = _rescore_vectorized(
                            v_cache, weight_vec,
                            body_form_filter=bff,
                            hymenium_filter=hf,
                            size_class_filter=scf,
                            recall_ks=recall_ks,
                            top_k=top_k,
                            alpha=a,
                        )
                        recall = {
                            k: hits[k] / total if total > 0 else 0.0 for k in recall_ks
                        }
                        weight_dict: dict[str, float | bool] = dict(weights_dict)
                        weight_dict["body_form_filter"] = bff
                        weight_dict["hymenium_filter"] = hf
                        weight_dict["size_class_filter"] = scf
                        weight_dict["alpha"] = a
                        results.append(SweepResult(
                            weights=weight_dict,
                            recall_at=recall,
                            hits_at=hits,
                            total=total,
                        ))

        if (idx + 1) % 500 == 0:
            logger.info("  Swept %d/%d configs...", idx + 1, n_configs)

    # Sort: primary recall@top_k desc, then recall@3, then recall@1
    sort_ks = sorted(recall_ks, reverse=True)
    results.sort(key=lambda r: tuple(r.recall_at.get(k, 0.0) for k in sort_ks), reverse=True)

    return results

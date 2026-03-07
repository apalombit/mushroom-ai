"""
Weight sweep optimizer: precompute per-group similarities once, then rescore
with thousands of weight configs as pure arithmetic.

A 1000-config sweep costs ~20s (precompute) + <1s (rescore) instead of hours.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from db.models import GroundTruthPair, ReconciledSpecies
from ingestion.rubric import _get_nested
from similarity.numeric import compute_numeric_similarity
from similarity.search import _GROUP_COLUMN, search_by_group
from similarity.weights import WEIGHT_FIELDS

logger = logging.getLogger(__name__)


@dataclass
class CandidateScores:
    scientific_name: str
    body_form_match: bool
    sims: dict[str, float]  # 7 keys matching WEIGHT_FIELDS


@dataclass
class QueryCache:
    query_name: str
    target_name: str
    candidates: list[CandidateScores]
    error: str | None = None


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
) -> list[QueryCache]:
    """
    Precompute per-group similarity scores for all ground truth pairs (both directions).

    Returns a list of QueryCache entries with raw similarity scores per candidate,
    independent of weight configuration.
    """
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

        if query_species.embedding_macro_visual is None:
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

            # Precompute body_form_match
            body_form_match = True
            if query_body_form:
                cand_body_form = _get_nested(cand_features, "overall_body_form")
                if cand_body_form and cand_body_form.lower() != query_body_form.lower():
                    body_form_match = False

            # Embedding similarities (6 groups)
            sims: dict[str, float] = {}
            for group in _GROUP_COLUMN:
                sims[group] = group_maps[group].get(sid, (None, 0.0))[1]

            # Numeric similarity
            sims["numeric"] = compute_numeric_similarity(query_features, cand_features)

            candidates.append(
                CandidateScores(
                    scientific_name=name,
                    body_form_match=body_form_match,
                    sims=sims,
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


def rescore(
    cache: list[QueryCache],
    weights: dict[str, float],
    body_form_filter: bool,
    recall_ks: tuple[int, ...] = (1, 3, 5),
    top_k: int = 5,
) -> SweepResult:
    """
    Pure arithmetic rescoring — no DB access. Called thousands of times.

    For each QueryCache entry: optionally filter by body_form_match, compute
    weighted overall score, sort, check if target in top-K.
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
            overall = sum(weights.get(f, 0.0) * cand.sims.get(f, 0.0) for f in WEIGHT_FIELDS)
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

    return SweepResult(weights=weight_dict, recall_at=recall, hits_at=hits, total=total)


def sweep_weights(
    session: Session,
    pairs: list[GroundTruthPair],
    n_configs: int = 200,
    recall_ks: tuple[int, ...] = (1, 3, 5),
    top_k: int = 5,
    pool_k: int = 60,
    seed: int = 42,
) -> list[SweepResult]:
    """
    Main entry point. Precomputes scores once, then sweeps random weight configs.

    Generates n_configs weight vectors via Dirichlet distribution, tests each
    with both body_form_filter=True and False (2 * n_configs total results).
    Returns results sorted by recall@top_k descending.
    """
    logger.info("Precomputing scores for %d pairs...", len(pairs))
    cache = precompute_scores(session, pairs, pool_k=pool_k)
    logger.info("Precompute done. Sweeping %d configs (x2 body_form_filter)...", n_configs)

    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(np.ones(len(WEIGHT_FIELDS)), n_configs)

    results: list[SweepResult] = []
    for sample in samples:
        weights = {f: float(sample[i]) for i, f in enumerate(WEIGHT_FIELDS)}
        for bff in (True, False):
            result = rescore(
                cache, weights, body_form_filter=bff, recall_ks=recall_ks, top_k=top_k
            )
            results.append(result)

    # Sort: primary recall@top_k desc, then recall@3, then recall@1
    sort_ks = sorted(recall_ks, reverse=True)
    results.sort(key=lambda r: tuple(r.recall_at.get(k, 0.0) for k in sort_ks), reverse=True)

    return results

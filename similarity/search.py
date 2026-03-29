"""
Similarity search engine.

Finds lookalike species by running independent pgvector similarity queries
per feature group (N embedding groups from the active profile), then merging
with numeric similarity, body-form gating, and tunable weights.

Query-time pipeline:
    1. Look up query species embeddings from ReconciledSpecies
    2. Run N pgvector cosine similarity queries (one per embedding group)
    3. Merge candidate sets, apply body-form filter
    4. Compute numeric similarity from features_json
    5. Compute weighted overall score (7 components)
    6. Apply user context (region/season) as boost
    7. Return top-K with per-group similarity breakdown
"""

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import settings
from db.models import ReconciledSpecies
from ingestion.rubric import (
    EMBEDDING_GROUPS,
    MORPHOLOGICAL_FIELDS,
    MORPHOLOGICAL_GROUP_NAMES,
    _get_nested,
)
from similarity.aggregation import (
    aggregate_blended,
    aggregate_contrastive_gate,
    aggregate_rrf,
    aggregate_weighted_avg,
)
from similarity.jaccard import compute_soft_jaccard
from similarity.numeric import compute_numeric_similarity
from similarity.weights import SimilarityWeights

logger = logging.getLogger(__name__)

_SAFE_EDIBILITY = frozenset({"edible", "choice", "conditionally edible"})
_DANGEROUS_EDIBILITY = frozenset({"toxic", "deadly", "inedible"})

# Maps group name → DB column, derived from the active profile
_GROUP_COLUMN = {g: f"embedding_{g}" for g in EMBEDDING_GROUPS}
_first_embed_col = f"embedding_{next(iter(EMBEDDING_GROUPS))}"


def search_by_group(
    session: Session,
    species_id: int,
    group: str,
    top_k: int = 20,
    taxon_group: str | None = None,
) -> list[tuple[int, str, float]]:
    """
    Run a pgvector cosine similarity query for one feature group.

    Returns list of (species_id, scientific_name, similarity_score),
    excluding the query species itself. Returns [] if the query species
    has no embedding for this group.

    If taxon_group is set, restricts candidates to the same taxonomic group.
    """
    col = _GROUP_COLUMN[group]
    group_clause = 'AND rs."group" = :taxon_group' if taxon_group else ""
    sql = text(f"""
        SELECT rs.id, rs.scientific_name,
               1 - (rs.{col} <=> q.{col}) AS similarity
        FROM reconciled_species rs,
             (SELECT {col} FROM reconciled_species WHERE id = :species_id) AS q
        WHERE rs.id != :species_id
          AND rs.{col} IS NOT NULL
          AND q.{col} IS NOT NULL
          {group_clause}
        ORDER BY rs.{col} <=> q.{col}
        LIMIT :top_k
    """)
    params: dict = {"species_id": species_id, "top_k": top_k}
    if taxon_group:
        params["taxon_group"] = taxon_group
    rows = session.execute(sql, params).fetchall()
    return [(row[0], row[1], float(row[2])) for row in rows]


def search_lookalikes(
    session: Session,
    species_name: str,
    weights: SimilarityWeights | None = None,
    top_k: int = 10,
    region: str | None = None,
    season: str | None = None,
    strategy: str = "weighted_avg",
    z_threshold: float = 0.0,
) -> tuple[ReconciledSpecies, list[dict[str, Any]]]:
    """
    Full lookalike search pipeline.

    Returns (query_species, candidates) where candidates is a list of dicts sorted
    by similarity_overall descending. Each dict contains:
        id, scientific_name, common_names, edibility, features_json,
        similarity_<group> for each group in the active profile,
        similarity_numeric, similarity_overall.

    Raises ValueError if species is not found or has no embeddings.
    """
    if weights is None:
        weights = SimilarityWeights()

    query_species = (
        session.query(ReconciledSpecies)
        .filter(ReconciledSpecies.scientific_name.ilike(species_name))
        .first()
    )
    if query_species is None:
        raise ValueError(f"Species not found: {species_name!r}")

    if getattr(query_species, _first_embed_col) is None:
        raise ValueError(f"Species {species_name!r} has no embeddings — run --embed first.")

    # Larger pool for merging (bump for learned_ranker to ensure adequate candidates)
    pool_k = max(top_k * 5, 100 if strategy == "learned_ranker" else 60)
    if weights.dangerous_filter:
        pool_k *= 2

    # Determine taxonomic group filter
    taxon_group = query_species.group if weights.group_filter and query_species.group else None

    # Run N embedding searches (with optional group pre-filter)
    group_maps: dict[str, dict[int, tuple[str, float]]] = {}
    for group in _GROUP_COLUMN:
        results = search_by_group(session, query_species.id, group, pool_k, taxon_group)
        group_maps[group] = {sid: (name, score) for sid, name, score in results}

    # Fallback: if group filter returned too few candidates, re-run without it
    if taxon_group:
        total_candidates = len({sid for gmap in group_maps.values() for sid in gmap})
        if total_candidates < pool_k:
            logger.info(
                "Group filter '%s' returned %d candidates (< %d), re-running unfiltered",
                taxon_group,
                total_candidates,
                pool_k,
            )
            for group in _GROUP_COLUMN:
                results = search_by_group(session, query_species.id, group, pool_k)
                group_maps[group] = {sid: (name, score) for sid, name, score in results}

    # Collect candidate IDs — morpho pool priority (Stage 2)
    morpho_ids: set[int] = set()
    for g in MORPHOLOGICAL_GROUP_NAMES:
        if g in group_maps:
            morpho_ids.update(group_maps[g])

    if weights.morpho_pool_required:
        # Candidates must appear in at least one morphological group
        all_ids = morpho_ids
    else:
        all_ids: set[int] = set()
        for gmap in group_maps.values():
            all_ids.update(gmap)

    # Rank maps for RRF: {group: {species_id: 1-based rank}}
    group_rank_maps: dict[str, dict[int, int]] = {}
    for group, gmap in group_maps.items():
        sorted_ids = sorted(gmap, key=lambda sid: gmap[sid][1], reverse=True)
        group_rank_maps[group] = {sid: rank + 1 for rank, sid in enumerate(sorted_ids)}

    # Group stats for contrastive gate: {group: (mean, std)}
    group_stats: dict[str, tuple[float, float]] = {}
    for group, gmap in group_maps.items():
        scores = [v[1] for v in gmap.values()]
        mean = sum(scores) / len(scores) if scores else 0.0
        variance = sum((s - mean) ** 2 for s in scores) / len(scores) if len(scores) > 1 else 0.0
        group_stats[group] = (mean, max(variance**0.5, 1e-6))

    query_features = query_species.features_json or {}
    query_body_form = _get_nested(query_features, "overall_body_form")
    query_hymenium = _get_nested(query_features, "hymenium.type")
    query_size_class = _get_nested(query_features, "overall_size_class")

    # Classify query edibility for dangerous-lookalikes filter
    _qe = (query_species.edibility or "").lower().strip()
    _qe_group = (
        "safe" if _qe in _SAFE_EDIBILITY
        else "dangerous" if _qe in _DANGEROUS_EDIBILITY
        else None
    )

    # Morphotype pre-filter (Stage 3) — filter candidate pool by signature match
    if weights.morphotype_prefilter:
        from similarity.morphotype import compute_morphotype_signature, morphotype_match_score

        query_sig = compute_morphotype_signature(query_features)
        if query_sig:
            min_match = settings.morphotype_min_match
            filtered: set[int] = set()
            for sid in all_ids:
                row = session.get(ReconciledSpecies, sid)
                sig = row.morphotype_signature if row else None
                if row and morphotype_match_score(query_sig, sig) >= min_match:
                    filtered.add(sid)
            if filtered:
                all_ids = filtered

    # Pre-load vocabulary for Jaccard scoring (when alpha < 1.0 or learned_ranker)
    vocab = None
    if weights.alpha < 1.0 or strategy == "learned_ranker":
        from ingestion.normalize import load_vocabulary

        vocab = load_vocabulary()

    candidates = []
    for sid in all_ids:
        # Get name from any group that found this candidate
        name = None
        for gmap in group_maps.values():
            if sid in gmap:
                name = gmap[sid][0]
                break

        species_row = session.get(ReconciledSpecies, sid)
        cand_features = (species_row.features_json if species_row else {}) or {}

        # Body-form gating
        if weights.body_form_filter and query_body_form:
            cand_body_form = _get_nested(cand_features, "overall_body_form")
            if cand_body_form and cand_body_form.lower() != query_body_form.lower():
                continue

        # Hymenium-type gating
        if weights.hymenium_filter and query_hymenium:
            cand_hymenium = _get_nested(cand_features, "hymenium.type")
            if cand_hymenium and cand_hymenium.lower() != query_hymenium.lower():
                continue

        # Size-class gating
        if weights.size_class_filter and query_size_class:
            cand_size = _get_nested(cand_features, "overall_size_class")
            if cand_size and cand_size.lower() != query_size_class.lower():
                continue

        # Dangerous-lookalikes gating: keep only opposite-edibility candidates
        if weights.dangerous_filter and _qe_group:
            ce = (species_row.edibility or "").lower().strip() if species_row else ""
            cg = (
                "safe" if ce in _SAFE_EDIBILITY
                else "dangerous" if ce in _DANGEROUS_EDIBILITY
                else None
            )
            if cg and cg == _qe_group:
                continue

        # Embedding similarities (6 groups)
        sims: dict[str, float] = {}
        for group in _GROUP_COLUMN:
            sims[group] = group_maps[group].get(sid, (None, 0.0))[1]

        # Numeric similarity
        sims["numeric"] = compute_numeric_similarity(query_features, cand_features)
        numeric_sim = sims["numeric"]

        # Overall score via selected aggregation strategy
        if strategy == "learned_ranker":
            sim_overall = 0.0  # placeholder — overwritten after batch scoring
        elif strategy == "rrf":
            sim_overall = aggregate_rrf(
                group_rank_maps, sid, len(group_maps), pool_k, weights.numeric, numeric_sim
            )
        elif strategy == "contrastive_gate":
            sim_overall = aggregate_contrastive_gate(
                {g: sims.get(g, 0.0) for g in group_maps},
                numeric_sim,
                group_stats,
                weights,
                z_threshold,
            )
        else:
            sim_overall = aggregate_weighted_avg(
                {**{g: sims.get(g, 0.0) for g in group_maps}, "numeric": numeric_sim},
                numeric_sim,
                weights,
            )

        # Jaccard similarity and blending (skip for learned_ranker — GBDT has its own features)
        jaccard_sim = 0.0
        if vocab is not None:
            jaccard_sim, _ = compute_soft_jaccard(query_features, cand_features, vocab)
            if strategy != "learned_ranker":
                sim_overall = aggregate_blended(sim_overall, jaccard_sim, weights.alpha)

        # Context boost for matching region/season
        boost = 0.0
        if species_row and cand_features:
            eco = cand_features.get("ecology", {})
            if region and any(
                region.lower() in r.lower() for r in (eco.get("geographic_regions") or [])
            ):
                boost += 0.02
            if season and season.lower() in [
                s.lower() for s in (eco.get("fruiting_seasons") or [])
            ]:
                boost += 0.02

        group_sim_keys = {f"similarity_{g}": round(sims[g], 4) for g in _GROUP_COLUMN}
        candidates.append(
            {
                "id": sid,
                "scientific_name": name,
                "common_names": (species_row.common_names if species_row else []) or [],
                "edibility": species_row.edibility if species_row else None,
                "features_json": cand_features,
                **group_sim_keys,
                "similarity_numeric": round(sims["numeric"], 4),
                "similarity_jaccard": round(jaccard_sim, 4),
                "similarity_overall": round(min(1.0, sim_overall + boost), 4),
            }
        )

    # Batch GBDT re-ranking: overwrite sim_overall with model predictions
    if strategy == "learned_ranker" and candidates:
        from similarity.ranker import score_candidates as ranker_score

        cand_features_list = [c["features_json"] for c in candidates]
        gbdt_scores = ranker_score(query_features, cand_features_list, vocab)
        for cand, score in zip(candidates, gbdt_scores):
            cand["similarity_overall"] = round(score, 4)

    candidates.sort(key=lambda x: x["similarity_overall"], reverse=True)
    return query_species, candidates[:top_k]


def build_comparison_table(
    query_species: ReconciledSpecies,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build per-feature comparison between query species and each candidate.

    Returns the candidates list with an added `feature_comparisons` key on each,
    containing a list of dicts with keys:
        feature_group, feature_name, query_value, candidate_value, is_similar.
    """
    query_features = query_species.features_json or {}

    # Use EMBEDDING_GROUPS for the 6 embedding groups + morphological for numeric fields
    all_fields: list[tuple[str, list[str]]] = [
        (name, fields) for name, fields in EMBEDDING_GROUPS.items()
    ]
    # Add numeric fields from MORPHOLOGICAL_FIELDS not covered by embedding groups
    embed_field_set = set()
    for fields in EMBEDDING_GROUPS.values():
        embed_field_set.update(fields)
    extra_morph = [f for f in MORPHOLOGICAL_FIELDS if f not in embed_field_set]
    if extra_morph:
        all_fields.append(("numeric", extra_morph))

    result = []
    for candidate in candidates:
        cand_features = candidate.get("features_json") or {}
        comparisons = []

        for group, field_list in all_fields:
            for field_path in field_list:
                q_val = _get_nested_str(query_features, field_path)
                c_val = _get_nested_str(cand_features, field_path)
                if q_val is None and c_val is None:
                    continue
                comparisons.append(
                    {
                        "feature_group": group,
                        "feature_name": field_path,
                        "query_value": q_val,
                        "candidate_value": c_val,
                        "is_similar": _values_similar(q_val, c_val),
                    }
                )

        result.append({**candidate, "feature_comparisons": comparisons})

    return result


def _get_nested_str(d: dict, path: str) -> str | None:
    """Walk dot-path into nested dict, return stringified value or None."""
    keys = path.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    if isinstance(current, list):
        joined = ", ".join(str(v) for v in current if v is not None)
        return joined if joined else None
    if isinstance(current, bool):
        return "yes" if current else "no"
    return str(current) if current is not None else None


def _values_similar(a: str | None, b: str | None) -> bool:
    """Heuristic: values are similar if they share at least one meaningful word."""
    if a is None or b is None:
        return False
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    meaningful = {w for w in a_words & b_words if len(w) > 1}
    return len(meaningful) > 0

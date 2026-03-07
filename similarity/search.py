"""
Similarity search engine.

Finds lookalike species by running independent pgvector similarity queries
per feature group (6 embedding groups), then merging with numeric similarity,
body-form gating, and tunable weights.

Query-time pipeline:
    1. Look up query species embeddings from ReconciledSpecies
    2. Run 6 pgvector cosine similarity queries (one per embedding group)
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

from db.models import ReconciledSpecies
from ingestion.rubric import EMBEDDING_GROUPS, MORPHOLOGICAL_FIELDS, _get_nested
from similarity.numeric import compute_numeric_similarity
from similarity.weights import WEIGHT_FIELDS, SimilarityWeights

logger = logging.getLogger(__name__)

# Maps weight field name → DB column for embedding-based groups
_GROUP_COLUMN = {
    "macro_visual": "embedding_macro_visual",
    "structural": "embedding_structural",
    "flesh_sensory": "embedding_flesh_sensory",
    "microscopic_lab": "embedding_microscopic_lab",
    "ecological": "embedding_ecological",
    "taxonomic": "embedding_taxonomic",
}


def search_by_group(
    session: Session,
    species_id: int,
    group: str,
    top_k: int = 20,
) -> list[tuple[int, str, float]]:
    """
    Run a pgvector cosine similarity query for one feature group.

    Returns list of (species_id, scientific_name, similarity_score),
    excluding the query species itself. Returns [] if the query species
    has no embedding for this group.
    """
    col = _GROUP_COLUMN[group]
    sql = text(f"""
        SELECT rs.id, rs.scientific_name,
               1 - (rs.{col} <=> q.{col}) AS similarity
        FROM reconciled_species rs,
             (SELECT {col} FROM reconciled_species WHERE id = :species_id) AS q
        WHERE rs.id != :species_id
          AND rs.{col} IS NOT NULL
          AND q.{col} IS NOT NULL
        ORDER BY rs.{col} <=> q.{col}
        LIMIT :top_k
    """)
    rows = session.execute(sql, {"species_id": species_id, "top_k": top_k}).fetchall()
    return [(row[0], row[1], float(row[2])) for row in rows]


def search_lookalikes(
    session: Session,
    species_name: str,
    weights: SimilarityWeights | None = None,
    top_k: int = 10,
    region: str | None = None,
    season: str | None = None,
) -> tuple[ReconciledSpecies, list[dict[str, Any]]]:
    """
    Full lookalike search pipeline.

    Returns (query_species, candidates) where candidates is a list of dicts sorted
    by similarity_overall descending. Each dict contains:
        id, scientific_name, common_names, edibility, features_json,
        similarity_macro_visual, similarity_structural, similarity_flesh_sensory,
        similarity_microscopic_lab, similarity_ecological, similarity_taxonomic,
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

    if query_species.embedding_macro_visual is None:
        raise ValueError(f"Species {species_name!r} has no embeddings — run --embed first.")

    # Larger pool for merging
    pool_k = max(top_k * 5, 60)

    # Run 6 embedding searches
    group_maps: dict[str, dict[int, tuple[str, float]]] = {}
    for group in _GROUP_COLUMN:
        results = search_by_group(session, query_species.id, group, pool_k)
        group_maps[group] = {sid: (name, score) for sid, name, score in results}

    # Collect all candidate IDs
    all_ids: set[int] = set()
    for gmap in group_maps.values():
        all_ids.update(gmap)

    query_features = query_species.features_json or {}
    query_body_form = _get_nested(query_features, "overall_body_form")

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

        # Embedding similarities (6 groups)
        sims: dict[str, float] = {}
        for group in _GROUP_COLUMN:
            sims[group] = group_maps[group].get(sid, (None, 0.0))[1]

        # Numeric similarity
        sims["numeric"] = compute_numeric_similarity(query_features, cand_features)

        # Weighted overall score
        sim_overall = sum(getattr(weights, f) * sims.get(f, 0.0) for f in WEIGHT_FIELDS)

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

        candidates.append(
            {
                "id": sid,
                "scientific_name": name,
                "common_names": (species_row.common_names if species_row else []) or [],
                "edibility": species_row.edibility if species_row else None,
                "features_json": cand_features,
                "similarity_macro_visual": round(sims["macro_visual"], 4),
                "similarity_structural": round(sims["structural"], 4),
                "similarity_flesh_sensory": round(sims["flesh_sensory"], 4),
                "similarity_microscopic_lab": round(sims["microscopic_lab"], 4),
                "similarity_ecological": round(sims["ecological"], 4),
                "similarity_taxonomic": round(sims["taxonomic"], 4),
                "similarity_numeric": round(sims["numeric"], 4),
                "similarity_overall": round(min(1.0, sim_overall + boost), 4),
            }
        )

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

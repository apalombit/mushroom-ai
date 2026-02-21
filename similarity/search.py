"""
Similarity search engine.

Finds lookalike species by running independent pgvector similarity queries
per feature group (morphological, ecological, taxonomic), then merging
and reranking with tunable weights.

Query-time pipeline:
    1. Look up query species embeddings from ReconciledSpecies
    2. Run 3 pgvector cosine similarity queries (one per group)
    3. Merge candidate sets, compute weighted score
    4. Apply user context (region/season) as boost
    5. Return top-K with per-group similarity breakdown
"""

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from db.models import ReconciledSpecies
from ingestion.rubric import ECOLOGICAL_FIELDS, MORPHOLOGICAL_FIELDS, TAXONOMIC_FIELDS
from similarity.weights import SimilarityWeights

logger = logging.getLogger(__name__)

_GROUP_COLUMN = {
    "morphological": "embedding_morphological",
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
        similarity_morphological, similarity_ecological, similarity_taxonomic,
        similarity_overall.

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

    if query_species.embedding_morphological is None:
        raise ValueError(
            f"Species {species_name!r} has no embeddings — run --embed first."
        )

    # Larger pool for merging — use 5x top_k (min 60) to ensure all candidates are
    # considered in databases up to a few hundred species.
    pool_k = max(top_k * 5, 60)
    morph_results = search_by_group(session, query_species.id, "morphological", pool_k)
    eco_results = search_by_group(session, query_species.id, "ecological", pool_k)
    taxon_results = search_by_group(session, query_species.id, "taxonomic", pool_k)

    # Index by species_id → (name, score)
    morph_map = {sid: (name, score) for sid, name, score in morph_results}
    eco_map = {sid: (name, score) for sid, name, score in eco_results}
    taxon_map = {sid: (name, score) for sid, name, score in taxon_results}

    all_ids = set(morph_map) | set(eco_map) | set(taxon_map)

    candidates = []
    for sid in all_ids:
        name = (morph_map.get(sid) or eco_map.get(sid) or taxon_map.get(sid))[0]
        sim_morph = morph_map.get(sid, (None, 0.0))[1]
        sim_eco = eco_map.get(sid, (None, 0.0))[1]
        sim_taxon = taxon_map.get(sid, (None, 0.0))[1]
        sim_overall = (
            weights.morphological * sim_morph
            + weights.ecological * sim_eco
            + weights.taxonomic * sim_taxon
        )

        # Context boost for matching region/season
        species_row = session.get(ReconciledSpecies, sid)
        boost = 0.0
        if species_row and species_row.features_json:
            eco = species_row.features_json.get("ecology", {})
            if region and any(
                region.lower() in r.lower()
                for r in (eco.get("geographic_regions") or [])
            ):
                boost += 0.02
            if season and season.lower() in [
                s.lower() for s in (eco.get("fruiting_seasons") or [])
            ]:
                boost += 0.02

        candidates.append({
            "id": sid,
            "scientific_name": name,
            "common_names": (species_row.common_names if species_row else []) or [],
            "edibility": species_row.edibility if species_row else None,
            "features_json": (species_row.features_json if species_row else {}) or {},
            "similarity_morphological": round(sim_morph, 4),
            "similarity_ecological": round(sim_eco, 4),
            "similarity_taxonomic": round(sim_taxon, 4),
            "similarity_overall": round(min(1.0, sim_overall + boost), 4),
        })

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

    all_fields = [
        ("morphological", MORPHOLOGICAL_FIELDS),
        ("ecological", ECOLOGICAL_FIELDS),
        ("taxonomic", TAXONOMIC_FIELDS),
    ]

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
                comparisons.append({
                    "feature_group": group,
                    "feature_name": field_path,
                    "query_value": q_val,
                    "candidate_value": c_val,
                    "is_similar": _values_similar(q_val, c_val),
                })

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
        text = ", ".join(str(v) for v in current if v is not None)
        return text if text else None
    if isinstance(current, bool):
        return "yes" if current else "no"
    return str(current) if current is not None else None


def _values_similar(a: str | None, b: str | None) -> bool:
    """Heuristic: values are similar if they share at least one meaningful word."""
    if a is None or b is None:
        return False
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    meaningful = {w for w in a_words & b_words if len(w) > 2}
    return len(meaningful) > 0

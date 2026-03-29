"""API routes for the Lookalikes Finder."""

import logging
import time

from fastapi import APIRouter, HTTPException

from api.schemas import (
    AssociationPair,
    FeatureComparison,
    LookalikeCandidate,
    LookalikeRequest,
    LookalikeResponse,
    SourceLink,
    SpeciesProfile,
    SpeciesSummary,
    normalize_image_refs,
)
from db.connection import get_session
from db.models import GroundTruthPair, ReconciledSpecies, SourceObservation
from ingestion.rubric import EMBEDDING_GROUPS
from similarity.explain import generate_explanation
from similarity.search import build_comparison_table, search_lookalikes
from similarity.weights import SimilarityWeights

router = APIRouter(prefix="/api/v1")
logger = logging.getLogger(__name__)


@router.post("/lookalikes", response_model=LookalikeResponse)
async def find_lookalikes(request: LookalikeRequest):
    """
    Find lookalike species for a given mushroom.

    Pipeline:
        1. Validate species exists and has embeddings
        2. Run per-group pgvector similarity queries
        3. Merge and rerank with weights
        4. Build feature comparison table
        5. Generate LLM explanation
        6. Return structured response
    """
    start = time.monotonic()
    weights = SimilarityWeights(
        weights=request.weights,
        numeric=request.weight_numeric,
        body_form_filter=request.body_form_filter,
        hymenium_filter=request.hymenium_filter,
        size_class_filter=request.size_class_filter,
        alpha=request.alpha,
        morpho_pool_required=request.morpho_pool_required,
        morphotype_prefilter=request.morphotype_prefilter,
        dangerous_filter=request.dangerous_filter,
    )

    session = get_session()
    try:
        total_species = session.query(ReconciledSpecies).count()

        try:
            query_species, candidates = search_lookalikes(
                session,
                species_name=request.species_name,
                weights=weights,
                top_k=request.top_k,
                region=request.region,
                season=request.season,
                strategy=request.aggregation_strategy,
                z_threshold=request.contrastive_z_threshold,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        comparison_table = build_comparison_table(query_species, candidates)

        # LLM explanation (non-fatal on failure)
        explanation_summary = None
        explanation_notable_pairs: list[str] = []
        explanation_safety_warning = None
        try:
            explanation = generate_explanation(request.species_name, comparison_table)
            explanation_summary = explanation.summary
            explanation_notable_pairs = explanation.notable_pairs
            explanation_safety_warning = explanation.safety_warning
        except Exception as e:
            logger.warning("Explanation generation failed: %s", e)

        # Batch-load image URLs for all candidates
        candidate_names = [c["scientific_name"] for c in comparison_table]
        image_rows = (
            session.query(
                ReconciledSpecies.scientific_name,
                ReconciledSpecies.image_urls,
            )
            .filter(ReconciledSpecies.scientific_name.in_(candidate_names))
            .all()
        )
        images_by_name = {
            r.scientific_name: normalize_image_refs(r.image_urls) for r in image_rows
        }

        # Batch-load source links for all candidates
        source_rows = (
            session.query(SourceObservation)
            .filter(SourceObservation.scientific_name.in_(candidate_names))
            .all()
        )
        sources_by_name: dict[str, list[SourceLink]] = {}
        for obs in source_rows:
            sources_by_name.setdefault(obs.scientific_name, []).append(
                SourceLink(source_name=obs.source_name, source_url=obs.source_url)
            )

        # Build response
        response_candidates = []
        for cand in comparison_table:
            feature_comps = [FeatureComparison(**fc) for fc in cand.get("feature_comparisons", [])]
            group_sims = {g: cand[f"similarity_{g}"] for g in EMBEDDING_GROUPS}
            response_candidates.append(
                LookalikeCandidate(
                    scientific_name=cand["scientific_name"],
                    common_names=cand.get("common_names") or [],
                    edibility=cand.get("edibility"),
                    group_similarities=group_sims,
                    similarity_numeric=cand["similarity_numeric"],
                    similarity_jaccard=cand.get("similarity_jaccard", 0.0),
                    similarity_overall=cand["similarity_overall"],
                    feature_comparisons=feature_comps,
                    image_urls=images_by_name.get(cand["scientific_name"], []),
                    sources=sources_by_name.get(cand["scientific_name"], []),
                )
            )

        latency = time.monotonic() - start
        _log_query_metrics(request.species_name, latency, response_candidates)

        return LookalikeResponse(
            query_species=request.species_name,
            query_species_edibility=query_species.edibility,
            query_species_image_urls=normalize_image_refs(query_species.image_urls),
            weights_used=weights.as_dict(),
            candidates=response_candidates,
            explanation_summary=explanation_summary,
            explanation_notable_pairs=explanation_notable_pairs,
            explanation_safety_warning=explanation_safety_warning,
            aggregation_strategy=request.aggregation_strategy,
            species_count_in_db=total_species,
        )
    finally:
        session.close()


@router.get("/species/summary", response_model=list[SpeciesSummary])
async def list_species_summary():
    """Lightweight species list: names + edibility only. No features or sources."""
    session = get_session()
    try:
        rows = (
            session.query(
                ReconciledSpecies.scientific_name,
                ReconciledSpecies.common_names,
                ReconciledSpecies.edibility,
            )
            .order_by(ReconciledSpecies.scientific_name)
            .all()
        )
        return [
            SpeciesSummary(
                scientific_name=r.scientific_name,
                common_names=r.common_names or [],
                edibility=r.edibility,
            )
            for r in rows
        ]
    finally:
        session.close()


@router.get("/species/{name}", response_model=SpeciesProfile)
async def get_species(name: str):
    """Get full species profile from the reconciled database."""
    session = get_session()
    try:
        row = (
            session.query(ReconciledSpecies)
            .filter(ReconciledSpecies.scientific_name.ilike(name))
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail=f"Species not found: {name!r}")
        return _species_profile(session, row)
    finally:
        session.close()


@router.get("/species", response_model=list[SpeciesProfile])
async def list_species(limit: int = 1000, offset: int = 0):
    """List all species in the database (paginated)."""
    session = get_session()
    try:
        rows = session.query(ReconciledSpecies).offset(offset).limit(limit).all()
        return [_species_profile(session, row) for row in rows]
    finally:
        session.close()


def _species_profile(session, row: ReconciledSpecies) -> SpeciesProfile:
    obs_rows = (
        session.query(SourceObservation)
        .filter(SourceObservation.scientific_name == row.scientific_name)
        .all()
    )
    sources = [
        SourceLink(source_name=obs.source_name, source_url=obs.source_url) for obs in obs_rows
    ]
    return SpeciesProfile(
        scientific_name=row.scientific_name,
        common_names=row.common_names or [],
        family=row.family,
        genus=row.genus,
        edibility=row.edibility,
        features=row.features_json or {},
        image_urls=normalize_image_refs(row.image_urls),
        source_count=row.source_count or 0,
        needs_review=row.needs_review or False,
        reconciliation_confidence=row.reconciliation_confidence,
        sources=sources,
    )


@router.get("/associations", response_model=list[AssociationPair])
async def list_associations():
    """List all known dangerous lookalike pairs from the ground truth dataset."""
    session = get_session()
    try:
        rows = session.query(GroundTruthPair).all()
        return [
            AssociationPair(
                species_a=r.species_a,
                species_b=r.species_b,
                danger_note=r.danger_note or None,
                source=r.source,
            )
            for r in rows
        ]
    finally:
        session.close()


def _log_query_metrics(
    species_name: str,
    latency: float,
    candidates: list[LookalikeCandidate],
) -> None:
    """Log query metrics to MLflow (best-effort)."""
    try:
        import mlflow

        with mlflow.start_run(run_name="lookalike_query"):
            mlflow.log_metric("query_latency_s", latency)
            if candidates:
                mlflow.log_metric("top_similarity", candidates[0].similarity_overall)
            mlflow.log_metric("candidates_returned", len(candidates))
            mlflow.log_param("query_species", species_name)
    except Exception:
        pass

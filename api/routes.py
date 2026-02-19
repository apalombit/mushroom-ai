"""API routes for the Lookalikes Finder."""

from fastapi import APIRouter, HTTPException

from api.schemas import LookalikeRequest, LookalikeResponse, SpeciesProfile

router = APIRouter(prefix="/api/v1")


@router.post("/lookalikes", response_model=LookalikeResponse)
async def find_lookalikes(request: LookalikeRequest):
    """
    Find lookalike species for a given mushroom.

    Pipeline:
        1. Validate species exists in database
        2. Retrieve its embeddings
        3. Run per-group pgvector similarity queries
        4. Merge and rerank with weights
        5. Build feature comparison table
        6. Generate LLM explanation
        7. Return structured response

    TODO:
        - [ ] Wire up similarity/search.py
        - [ ] Wire up similarity/explain.py
        - [ ] Add MLflow query logging
    """
    raise NotImplementedError("Implement lookalike pipeline")


@router.get("/species/{name}", response_model=SpeciesProfile)
async def get_species(name: str):
    """Get full species profile from the reconciled database."""
    # TODO: Query ReconciledSpecies by scientific_name or common_name
    raise NotImplementedError("Implement species lookup")


@router.get("/species", response_model=list[SpeciesProfile])
async def list_species(limit: int = 100, offset: int = 0):
    """List all species in the database (paginated)."""
    # TODO: Query ReconciledSpecies with pagination
    raise NotImplementedError("Implement species listing")

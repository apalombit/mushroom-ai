"""
API request/response schemas.

These are the public contract — what users send and receive.
Separate from llm/schemas.py (LLM extraction contracts) and db/models.py (storage).
"""

from pydantic import BaseModel, Field


class LookalikeRequest(BaseModel):
    species_name: str = Field(..., examples=["Amanita caesarea"])
    region: str | None = Field(None, examples=["Northern Italy"])
    season: str | None = Field(None, examples=["autumn"])

    # Optional weight overrides (0-1, will be normalized)
    weight_macro_visual: float | None = None
    weight_structural: float | None = None
    weight_flesh_sensory: float | None = None
    weight_microscopic_lab: float | None = None
    weight_ecological: float | None = None
    weight_taxonomic: float | None = None
    weight_numeric: float | None = None
    body_form_filter: bool | None = None

    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")


class FeatureComparison(BaseModel):
    """Per-feature comparison between query species and a lookalike candidate."""

    feature_group: str  # "morphological", "ecological", "taxonomic"
    feature_name: str  # e.g. "cap.shape", "ecology.habitat_types"
    query_value: str | None
    candidate_value: str | None
    is_similar: bool  # Whether this feature contributed to the match


class LookalikeCandidate(BaseModel):
    """A single lookalike result with similarity breakdown."""

    scientific_name: str
    common_names: list[str] = []
    edibility: str | None = None

    # Per-group similarity scores (0-1, higher = more similar)
    similarity_macro_visual: float
    similarity_structural: float
    similarity_flesh_sensory: float
    similarity_microscopic_lab: float
    similarity_ecological: float
    similarity_taxonomic: float
    similarity_numeric: float
    similarity_overall: float  # Weighted combination

    # Feature-level comparison
    feature_comparisons: list[FeatureComparison] = []


class LookalikeResponse(BaseModel):
    """Full response for a lookalike query."""

    query_species: str
    query_species_edibility: str | None = None
    weights_used: dict[str, float]

    candidates: list[LookalikeCandidate]

    # LLM-generated explanation
    explanation_summary: str | None = None
    explanation_notable_pairs: list[str] = []
    explanation_safety_warning: str | None = None

    species_count_in_db: int  # How many species were compared


class SourceLink(BaseModel):
    """A source that contributed to a reconciled species profile."""

    source_name: str
    source_url: str | None = None


class SpeciesProfile(BaseModel):
    """Full species profile from the reconciled database."""

    scientific_name: str
    common_names: list[str] = []
    family: str | None = None
    genus: str | None = None
    edibility: str | None = None
    features: dict  # Full features_json
    source_count: int = 0
    needs_review: bool = False
    reconciliation_confidence: float | None = None
    sources: list[SourceLink] = []


class AssociationPair(BaseModel):
    """A known dangerous lookalike pair from the ground truth dataset."""

    species_a: str
    species_b: str
    danger_note: str | None = None
    source: str | None = None

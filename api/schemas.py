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
    weights: dict[str, float] | None = None  # {group_name: weight}
    weight_numeric: float | None = None
    body_form_filter: bool | None = None
    hymenium_filter: bool | None = None
    size_class_filter: bool | None = None
    dangerous_filter: bool | None = None

    # Embedding vs Jaccard blend: 1.0 = pure embedding, 0.0 = pure Jaccard
    alpha: float | None = None
    morpho_pool_required: bool | None = None
    morphotype_prefilter: bool | None = None

    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")

    aggregation_strategy: str = "weighted_avg"
    contrastive_z_threshold: float = 0.0


class FeatureComparison(BaseModel):
    """Per-feature comparison between query species and a lookalike candidate."""

    feature_group: str  # "morphological", "ecological", "taxonomic"
    feature_name: str  # e.g. "cap.shape", "ecology.habitat_types"
    query_value: str | None
    candidate_value: str | None
    is_similar: bool  # Whether this feature contributed to the match


class ImageRef(BaseModel):
    """An image with source attribution."""

    image_url: str
    source_name: str
    source_url: str | None = None


def normalize_image_refs(raw: list | None) -> list[ImageRef]:
    """Convert legacy list[str] or new list[dict] to list[ImageRef]."""
    if not raw:
        return []
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append(ImageRef(image_url=item, source_name="unknown"))
        elif isinstance(item, dict):
            result.append(ImageRef(**item))
    return result


class SourceLink(BaseModel):
    """A source that contributed to a reconciled species profile."""

    source_name: str
    source_url: str | None = None


class LookalikeCandidate(BaseModel):
    """A single lookalike result with similarity breakdown."""

    scientific_name: str
    common_names: list[str] = []
    edibility: str | None = None

    # Per-group similarity scores (0-1, higher = more similar)
    group_similarities: dict[str, float]  # {group_name: score}
    similarity_numeric: float
    similarity_jaccard: float = 0.0
    similarity_overall: float  # Weighted combination

    # Feature-level comparison
    feature_comparisons: list[FeatureComparison] = []

    # Reference images with source attribution
    image_urls: list[ImageRef] = []

    # Reference links to external sources
    sources: list[SourceLink] = []


class LookalikeResponse(BaseModel):
    """Full response for a lookalike query."""

    query_species: str
    query_species_edibility: str | None = None
    query_species_image_urls: list[ImageRef] = []
    weights_used: dict[str, float]

    candidates: list[LookalikeCandidate]

    # LLM-generated explanation
    explanation_summary: str | None = None
    explanation_notable_pairs: list[str] = []
    explanation_safety_warning: str | None = None

    aggregation_strategy: str  # echoes what was used
    species_count_in_db: int  # How many species were compared


class SpeciesProfile(BaseModel):
    """Full species profile from the reconciled database."""

    scientific_name: str
    common_names: list[str] = []
    family: str | None = None
    genus: str | None = None
    edibility: str | None = None
    features: dict  # Full features_json
    image_urls: list[ImageRef] = []
    source_count: int = 0
    needs_review: bool = False
    reconciliation_confidence: float | None = None
    sources: list[SourceLink] = []


class SpeciesSummary(BaseModel):
    """Lightweight species listing: name + edibility only."""

    scientific_name: str
    common_names: list[str] = []
    edibility: str | None = None


class AssociationPair(BaseModel):
    """A known dangerous lookalike pair from the ground truth dataset."""

    species_a: str
    species_b: str
    danger_note: str | None = None
    source: str | None = None

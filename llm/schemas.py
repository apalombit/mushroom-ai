"""
Pydantic schemas for LLM-structured outputs.

Three categories:
    1. Extraction — used at ingestion time to parse source text into features
    2. Reconciliation — used to merge multiple source observations
    3. Explanation — used at query time to generate human-readable summaries

These are LLM output contracts (Instructor validates against them).
API response schemas live in api/schemas.py.
"""

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Feature extraction schemas (ingestion time)
# ---------------------------------------------------------------------------


class ExtractedCapFeatures(BaseModel):
    """Cap (pileus) features extracted from a source description."""

    shape: str | None = Field(None, description="e.g. convex, flat, umbonate, conical, bell-shaped")
    colors: list[str] = Field(default_factory=list, description="Observed color(s), e.g. ['orange', 'red-orange']")
    surface_texture: str | None = Field(None, description="e.g. smooth, scaly, sticky, dry, striate")
    scales_or_warts: str | None = Field(None, description="Describe any scales, warts, or patches on cap surface")
    diameter_min_cm: float | None = Field(None, description="Minimum typical cap diameter in cm")
    diameter_max_cm: float | None = Field(None, description="Maximum typical cap diameter in cm")


class ExtractedGillFeatures(BaseModel):
    """Gill/pore/teeth features."""

    hymenium_type: str | None = Field(None, description="gills, pores, teeth, smooth, or ridges")
    attachment: str | None = Field(None, description="free, adnate, adnexed, decurrent, sinuate")
    spacing: str | None = Field(None, description="crowded, close, distant")
    color: str | None = Field(None, description="Gill/pore color at maturity")
    color_young: str | None = Field(None, description="Gill/pore color when young, if different")


class ExtractedStemFeatures(BaseModel):
    """Stem (stipe) features."""

    height_min_cm: float | None = Field(None, description="Minimum typical height in cm")
    height_max_cm: float | None = Field(None, description="Maximum typical height in cm")
    diameter_min_cm: float | None = Field(None, description="Minimum typical diameter in cm")
    diameter_max_cm: float | None = Field(None, description="Maximum typical diameter in cm")
    color: str | None = None
    surface_texture: str | None = Field(None, description="e.g. smooth, fibrous, reticulate")
    consistency: str | None = Field(None, description="e.g. solid, hollow, fibrous, brittle, fleshy")


class ExtractedVeilFeatures(BaseModel):
    """Veil / ring (annulus) features — independent from stem."""

    present: bool | None = Field(None, description="Whether a ring/veil is present")
    type: str | None = Field(None, description="e.g. ring-like, partial veil, skirt, cortina")
    shape: str | None = Field(None, description="e.g. pendant, ascending, flaring")
    mobility: str | None = Field(None, description="e.g. fixed, movable/sliding")
    color: str | None = None
    brittleness: str | None = Field(None, description="e.g. fragile, persistent, membranous")


class ExtractedVolvaFeatures(BaseModel):
    """Volva features — independent from stem."""

    present: bool | None = Field(None, description="Whether a volva is present at stem base")
    type: str | None = Field(None, description="e.g. sac-like, sheathing, friable remnants")
    shape: str | None = Field(None, description="e.g. cup-shaped, bag-like, ridged")
    color: str | None = None


class ExtractedFleshFeatures(BaseModel):
    """Flesh (context) features."""

    color: str | None = None
    bruising_color: str | None = Field(None, description="Color change when cut or bruised, if any")
    odor: str | None = Field(None, description="Distinctive odor if any: mushroomy, anise, foul, mealy, etc.")


class ExtractedEcologicalFeatures(BaseModel):
    """Where and when the species grows."""

    habitat_types: list[str] = Field(default_factory=list, description="e.g. deciduous forest, coniferous forest, grassland")
    substrate: str | None = Field(None, description="e.g. soil, wood, dung, leaf litter, living tree")
    associated_trees: list[str] = Field(default_factory=list, description="Tree genera for mycorrhizal species: oak, birch, pine...")
    fruiting_seasons: list[str] = Field(default_factory=list, description="e.g. ['late summer', 'autumn']")
    geographic_regions: list[str] = Field(default_factory=list, description="e.g. ['Europe', 'North America']")
    altitude_notes: str | None = Field(None, description="Altitude range or preference if known")
    growth_pattern: str | None = Field(None, description="solitary, clustered, fairy ring, trooping")
    growth_position: str | None = Field(None, description="ground level, on fallen wood, on living trees, on stumps")


class ExtractedSpeciesFeatures(BaseModel):
    """
    Full feature extraction for one species from one source.

    This is the schema Instructor validates against when parsing source text.
    One instance per species per source → stored as Layer 1.
    """

    scientific_name: str = Field(description="Binomial name, e.g. 'Amanita caesarea'")
    common_names: list[str] = Field(default_factory=list, description="Common names in any language")
    family: str | None = None
    genus: str | None = None

    # Morphological
    cap: ExtractedCapFeatures = Field(default_factory=ExtractedCapFeatures)
    gills: ExtractedGillFeatures = Field(default_factory=ExtractedGillFeatures)
    stem: ExtractedStemFeatures = Field(default_factory=ExtractedStemFeatures)
    veil: ExtractedVeilFeatures = Field(default_factory=ExtractedVeilFeatures)
    volva: ExtractedVolvaFeatures = Field(default_factory=ExtractedVolvaFeatures)
    flesh: ExtractedFleshFeatures = Field(default_factory=ExtractedFleshFeatures)
    spore_print_color: str | None = None
    overall_size_class: str | None = Field(None, description="small, medium, large")

    # Ecological
    ecology: ExtractedEcologicalFeatures = Field(default_factory=ExtractedEcologicalFeatures)

    # Safety
    edibility: str | None = Field(None, description="edible, conditionally edible, inedible, toxic, deadly")
    known_toxins: list[str] = Field(default_factory=list)
    known_lookalikes: list[str] = Field(
        default_factory=list,
        description="Species names mentioned as lookalikes in the source text",
    )

    # Extraction metadata
    extraction_notes: str | None = Field(
        None,
        description="Anything ambiguous or uncertain in the source text",
    )


# ---------------------------------------------------------------------------
# 2. Reconciliation schemas (merging multiple sources)
# ---------------------------------------------------------------------------


class ReconciliationResult(BaseModel):
    """
    LLM output when reconciling multiple source observations for one species.

    The LLM receives all Layer 1 entries for a species and produces the
    canonical Layer 2 profile. Fields that cannot be confidently reconciled
    are flagged for human review.
    """

    reconciled_features: ExtractedSpeciesFeatures = Field(
        description="The canonical merged feature profile for this species"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the reconciled profile (0-1)",
    )
    conflicts: list[str] = Field(
        default_factory=list,
        description="List of fields where sources disagreed and reconciliation was uncertain",
    )
    needs_review: bool = Field(
        description="True if any conflicts require human review",
    )
    review_notes: str | None = Field(
        None,
        description="Explanation of what needs human attention and why",
    )


# ---------------------------------------------------------------------------
# 3. Explanation schemas (query time)
# ---------------------------------------------------------------------------


class LookalikeExplanation(BaseModel):
    """LLM-generated explanation of why species are lookalikes."""

    summary: str = Field(
        description="2-3 sentence 'at a glance' summary of the most notable "
        "lookalikes and the key reason they cause confusion."
    )
    notable_pairs: list[str] = Field(
        description="The 2-3 most important lookalike relationships to highlight, "
        "each as a concise sentence explaining what makes them confusable "
        "and what the critical distinguishing feature is."
    )
    safety_warning: str = Field(
        description="Safety note if any lookalikes are toxic or deadly. "
        "Must be prominent and unambiguous."
    )

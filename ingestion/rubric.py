"""
Feature rubric: defines the controlled vocabularies and feature groups
used throughout the system.

This rubric governs:
    - What the ingestion LLM extracts (via schemas in llm/schemas.py)
    - How features are grouped for embedding (morphological, ecological, taxonomic)
    - What the similarity engine compares

The rubric is designed to be expanded over time. Adding new features
requires updating: this file, the extraction schema, and the embedding logic.
"""

# ---------------------------------------------------------------------------
# Feature groups — define which fields belong to each similarity group
# ---------------------------------------------------------------------------

MORPHOLOGICAL_FIELDS = [
    "cap.shape", "cap.colors", "cap.surface_texture", "cap.scales_or_warts",
    "cap.diameter_min_cm", "cap.diameter_max_cm",
    "gills.hymenium_type", "gills.attachment", "gills.spacing", "gills.color",
    "stem.color", "stem.surface_texture", "stem.consistency",
    "stem.height_min_cm", "stem.height_max_cm",
    "veil.present", "veil.type", "veil.shape", "veil.color",
    "volva.present", "volva.type", "volva.shape", "volva.color",
    "flesh.color", "flesh.bruising_color", "flesh.odor",
    "spore_print_color",
    "overall_size_class",
]

ECOLOGICAL_FIELDS = [
    "ecology.habitat_types", "ecology.substrate", "ecology.associated_trees",
    "ecology.fruiting_seasons", "ecology.geographic_regions",
    "ecology.altitude_notes",
    "ecology.growth_pattern", "ecology.growth_position",
]

TAXONOMIC_FIELDS = [
    "family", "genus",
]


def features_to_text(features: dict, field_list: list[str]) -> str:
    """
    Convert a subset of species features into a natural language description
    suitable for embedding.

    This is the bridge between structured data and vector embeddings.
    The text should read like a concise field guide entry for that feature group.

    Args:
        features: The full features_json dict from a ReconciledSpecies row.
        field_list: Which fields to include (e.g., MORPHOLOGICAL_FIELDS).

    Returns:
        A natural language string describing those features.

    TODO:
        - [ ] Implement field extraction from nested dict
        - [ ] Generate readable text (not just "key: value" dumps)
        - [ ] Handle missing fields gracefully (skip, don't say "None")
    """
    raise NotImplementedError("Implement feature-to-text conversion")

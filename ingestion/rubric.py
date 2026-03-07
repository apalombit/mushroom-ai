"""
Feature rubric: defines the controlled vocabularies and feature groups
used throughout the system.

This rubric governs:
    - What the ingestion LLM extracts (via schemas in llm/schemas.py)
    - How features are grouped for embedding (6 embedding groups + numeric)
    - What the similarity engine compares

The rubric is designed to be expanded over time. Adding new features
requires updating: this file, the extraction schema, and the embedding logic.
"""

# ---------------------------------------------------------------------------
# Morphological sub-groups — focused embedding groups for better recall
# ---------------------------------------------------------------------------

# What you see at a glance: cap, hymenium type, body form, size, spore print
MACRO_VISUAL_FIELDS = [
    "cap.shape",
    "cap.colors",
    "cap.color_faded",
    "cap.color_pattern",
    "cap.surface_texture",
    "cap.surface_moisture",
    "cap.scales_or_warts",
    "cap.margin_type",
    "cap.margin_lined_at_maturity",
    "cap.bruising_color",
    "cap.central_depression",
    "hymenium.type",
    "spore_print_color",
    "overall_size_class",
    "overall_body_form",
    "growth_habit",
    "edibility_status",
    "known_lookalikes",
]

# Secondary visual cues: gills/pores (non-numeric), stem (non-numeric), veil, volva
STRUCTURAL_FIELDS = [
    # Gills
    "gills.attachment",
    "gills.spacing",
    "gills.color",
    "gills.color_with_age",
    "gills.thickness",
    "gills.texture",
    "gills.edge_texture",
    # Pores (non-numeric)
    "pores.color",
    "pores.color_with_age",
    "pores.bruising_color",
    # Stem (non-numeric)
    "stem.color",
    "stem.color_with_age",
    "stem.surface_texture",
    "stem.reticulation",
    "stem.shape",
    "stem.attachment_position",
    "stem.consistency",
    "stem.hollow_or_solid",
    "stem.base_color",
    "stem.basal_mycelium_color",
    "stem.finger_stain_color",
    "stem.bruising_color",
    # Veil
    "veil.present",
    "veil.type",
    "veil.cortina_present",
    "veil.shape",
    "veil.color",
    "veil.ring_position",
    "veil.ring_mobility",
    "veil.ring_persistence",
    # Volva
    "volva.present",
    "volva.type",
    "volva.shape",
    "volva.color",
]

# Handling features: flesh color, bruising, odor, taste, latex, texture
FLESH_SENSORY_FIELDS = [
    "flesh.color",
    "flesh.bruising_color",
    "flesh.latex",
    "flesh.odor",
    "flesh.taste",
    "flesh.texture",
    "flesh.hyphal_structure",
    "flesh.cap_stem_consistency",
    "flesh.quantity",
]

# Lab confirmation: spore morphology, cystidia, pileipellis, chemical reactions
MICROSCOPIC_LAB_FIELDS = [
    # Spore (non-numeric)
    "spore.shape",
    "spore.ornamentation",
    "spore.amyloidity",
    "spore.color_in_KOH",
    # Basidia
    "microscopic.basidia_spore_count",
    # Cystidia
    "microscopic.cheilocystidia_shape",
    "microscopic.cheilocystidia_dims_um",
    "microscopic.pleurocystidia_shape",
    "microscopic.pleurocystidia_dims_um",
    "microscopic.cystidia_color_in_KOH",
    # Pileipellis (non-numeric)
    "microscopic.pileipellis_type",
    "microscopic.pileipellis_terminal_cell_shape",
    # Chemical reactions
    "chemical.KOH_cap",
    "chemical.KOH_flesh",
    "chemical.NH4OH_cap",
    "chemical.NH4OH_flesh",
    "chemical.FeSO4_cap",
    "chemical.FeSO4_flesh",
]

# ---------------------------------------------------------------------------
# Ecological and taxonomic groups (unchanged)
# ---------------------------------------------------------------------------

ECOLOGICAL_FIELDS = [
    "ecology.trophic_mode",  # mycorrhizal | saprotrophic | parasitic
    "ecology.habitat_types",  # hardwood forest, conifer forest, grassland, etc.
    "ecology.substrate",  # soil, wood, dung, leaf litter, etc.
    "ecology.associated_trees",  # oak, beech, pine, etc.
    "ecology.fruiting_seasons",  # spring | summer | fall | winter
    "ecology.fruiting_months",  # e.g. "July–September", "late spring and summer"
    "ecology.geographic_regions",
    "ecology.altitude_notes",
    "ecology.growth_position",  # terrestrial | lignicolous | coprophilous | etc.
    "ecology.microhabitat_notes",  # e.g. mossy ground, disturbed areas
]

TAXONOMIC_FIELDS = [
    "kingdom",
    "phylum",  # e.g. Basidiomycetes — useful for broad filtering
    "order",
    "family",
    "genus",
    "species",
    "common_names",
    "synonyms",
]

# ---------------------------------------------------------------------------
# Numeric fields — for direct comparison, not embedded
# ---------------------------------------------------------------------------

# Range pairs: (min_field, max_field)
NUMERIC_RANGE_FIELDS: list[tuple[str, str]] = [
    ("cap.diameter_min_cm", "cap.diameter_max_cm"),
    ("stem.height_min_cm", "stem.height_max_cm"),
    ("stem.diameter_min_cm", "stem.diameter_max_cm"),
    ("spore.length_min_um", "spore.length_max_um"),
    ("spore.width_min_um", "spore.width_max_um"),
]

# Single values: (field, tolerance) — tolerance is max distance for score = 0
NUMERIC_SINGLE_FIELDS: list[tuple[str, float]] = [
    ("pores.density_per_mm", 3.0),
    ("tubes.depth_mm", 15.0),
    ("spore.spine_length_um", 2.0),
    ("spore.spine_base_width_um", 1.0),
    ("microscopic.pileipellis_element_width_um", 5.0),
]

# Combined list for iteration
NUMERIC_FIELDS: list[tuple] = NUMERIC_RANGE_FIELDS + NUMERIC_SINGLE_FIELDS

# ---------------------------------------------------------------------------
# Embedding groups — each maps to one pgvector column
# ---------------------------------------------------------------------------

EMBEDDING_GROUPS: dict[str, list[str]] = {
    "macro_visual": MACRO_VISUAL_FIELDS,
    "structural": STRUCTURAL_FIELDS,
    "flesh_sensory": FLESH_SENSORY_FIELDS,
    "microscopic_lab": MICROSCOPIC_LAB_FIELDS,
    "ecological": ECOLOGICAL_FIELDS,
    "taxonomic": TAXONOMIC_FIELDS,
}

# ---------------------------------------------------------------------------
# Backward-compatible union — used by build_comparison_table
# ---------------------------------------------------------------------------

# All numeric field paths (flattened from range pairs + single values)
_NUMERIC_FIELD_PATHS = [f for pair in NUMERIC_RANGE_FIELDS for f in pair] + [
    f for f, _ in NUMERIC_SINGLE_FIELDS
]

MORPHOLOGICAL_FIELDS = (
    MACRO_VISUAL_FIELDS
    + STRUCTURAL_FIELDS
    + FLESH_SENSORY_FIELDS
    + MICROSCOPIC_LAB_FIELDS
    + _NUMERIC_FIELD_PATHS
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nested(d: dict, path: str):
    """Walk a dot-separated path into a nested dict. Returns None if any key is missing."""
    keys = path.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def features_to_text(features: dict, field_list: list[str]) -> str:
    """
    Convert a subset of species features into a natural language description
    suitable for embedding.

    Walks each dot-path field into the nested features dict, skips None/empty
    values, and joins non-null values into concise field-guide prose.

    Args:
        features: The full features_json dict from a ReconciledSpecies row.
        field_list: Which fields to include (e.g., MORPHOLOGICAL_FIELDS).

    Returns:
        A natural language string describing those features.
    """
    parts = []
    for field_path in field_list:
        value = _get_nested(features, field_path)
        if value is None or value == "" or value == []:
            continue

        label = field_path.split(".")[-1].replace("_", " ")

        if isinstance(value, list):
            text = ", ".join(str(v) for v in value if v is not None)
            if text:
                parts.append(f"{label}: {text}")
        elif isinstance(value, bool):
            parts.append(f"{label}: {'yes' if value else 'no'}")
        else:
            parts.append(f"{label}: {value}")

    return ". ".join(parts) + ("." if parts else "")

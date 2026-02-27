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
    # --- Cap ---
    "cap.shape",                        # convex, broadly convex, flat, depressed, umbonate, etc.
    "cap.colors",                       # list of observed colors (fresh)
    "cap.color_faded",                  # color when dried/faded — e.g. buff, brownish
    "cap.hygrophanous",                 # bool — changes color markedly with moisture loss
    "cap.color_pattern",                # uniform, darker center, two-toned, mottled, etc.
    "cap.surface_texture",              # velvety, slimy, smooth, fibrous, hairy-scaly, dry, etc.
    "cap.scales_or_warts",              # present/absent + description
    "cap.margin_type",                  # inrolled, wavy, even, striate/lined, etc.
    "cap.margin_lined_at_maturity",     # bool — lined/striate at margin with age (e.g. Laccaria)
    "cap.central_depression",           # bool — depressed at disc
    "cap.diameter_min_cm",
    "cap.diameter_max_cm",

    # --- Hymenium type ---
    "hymenium.type",                    # gills | pores | teeth | ridges | smooth

    # --- Gills (if hymenium.type == gills) ---
    "gills.attachment",                 # free, adnate, decurrent, sinuate, etc.
    "gills.spacing",                    # crowded, close, subdistant, distant
    "gills.color",
    "gills.color_with_age",
    "gills.thickness",                  # thin, thick — e.g. Laccaria thick gills
    "gills.texture",                    # waxy, brittle, normal

    # --- Pores / Tubes (if hymenium.type == pores) ---
    "pores.color",
    "pores.color_with_age",
    "pores.bruising_color",             # e.g. slowly orangish-brown, blue, none
    "pores.density_per_mm",
    "tubes.depth_mm",

    # --- Stem ---
    "stem.color",
    "stem.color_with_age",
    "stem.surface_texture",             # smooth, reticulate, fibrous, hairy, scaly, powdery
    "stem.reticulation",                # none | partial | full — key for boletes
    "stem.shape",                       # equal, club-shaped, tapered base, bulbous, swollen base
    "stem.consistency",                 # firm, fibrous, spongy, brittle
    "stem.hollow_or_solid",             # hollow | stuffed | solid
    "stem.base_color",
    "stem.basal_mycelium_color",        # color of mycelium threads at base — e.g. lilac in Laccaria
    "stem.finger_stain_color",          # e.g. yellow stain from Retiboletus ornatipes
    "stem.height_min_cm",
    "stem.height_max_cm",
    "stem.diameter_min_cm",
    "stem.diameter_max_cm",

    # --- Veil ---
    "veil.present",
    "veil.type",                        # partial | universal | cortina | absent
    "veil.cortina_present",             # bool — explicit flag; key differentiator vs. Cortinarius
    "veil.shape",
    "veil.color",

    # --- Volva ---
    "volva.present",
    "volva.type",
    "volva.shape",
    "volva.color",

    # --- Flesh ---
    "flesh.color",
    "flesh.bruising_color",
    "flesh.odor",
    "flesh.taste",                      # mild, bitter, acrid, farinaceous, not distinctive
    "flesh.texture",                    # firm, soft, brittle, insubstantial, watery
    "flesh.quantity",                   # insubstantial | thin | moderate | thick

    # --- Spore print ---
    "spore_print_color",

    # --- Spores (microscopic) ---
    "spore.shape",                      # globose, ellipsoid, subfusoid, amygdaliform, etc.
    "spore.length_min_um",
    "spore.length_max_um",
    "spore.width_min_um",
    "spore.width_max_um",
    "spore.ornamentation",              # smooth | echinulate | warty | reticulate | striate
    "spore.spine_length_um",            # for echinulate spores — e.g. 1.5–3 µm in Laccaria
    "spore.spine_base_width_um",        # diagnostic detail for Laccaria genus
    "spore.amyloidity",                 # amyloid | inamyloid | dextrinoid
    "spore.color_in_KOH",

    # --- Microscopic: basidia ---
    "microscopic.basidia_spore_count",  # 4-spored, 2-spored, mixed — can differ within species

    # --- Microscopic: cystidia ---
    "microscopic.cheilocystidia_shape",       # shape descriptor — narrowly cylindric, subclavate, etc.
    "microscopic.cheilocystidia_dims_um",     # e.g. "25–65 x 4–12"
    "microscopic.pleurocystidia_shape",
    "microscopic.pleurocystidia_dims_um",
    "microscopic.cystidia_color_in_KOH",

    # --- Microscopic: pileipellis ---
    "microscopic.pileipellis_type",           # cutis | trichoderm | ixocutis | hymeniderm
    "microscopic.pileipellis_element_width_um",
    "microscopic.pileipellis_terminal_cell_shape",  # subclavate, capitate, rounded, etc.

    # --- Chemical reactions ---
    "chemical.KOH_cap",
    "chemical.KOH_flesh",
    "chemical.NH4OH_cap",
    "chemical.NH4OH_flesh",
    "chemical.FeSO4_cap",
    "chemical.FeSO4_flesh",

    # --- Overall ---
    "overall_size_class",               # small | medium | large
    "edibility_status",                 # edible | inedible | toxic | choice | unknown
    "known_lookalikes",
]

ECOLOGICAL_FIELDS = [
    "ecology.trophic_mode",             # mycorrhizal | saprotrophic | parasitic
    "ecology.habitat_types",            # hardwood forest, conifer forest, grassland, etc.
    "ecology.substrate",                # soil, wood, dung, leaf litter, etc.
    "ecology.associated_trees",         # oak, beech, pine, etc.
    "ecology.fruiting_seasons",         # spring | summer | fall | winter
    "ecology.fruiting_months",          # e.g. "July–September", "late spring and summer"
    "ecology.geographic_regions",
    "ecology.altitude_notes",
    "ecology.growth_pattern",           # solitary | scattered | gregarious | clustered
    "ecology.growth_position",          # terrestrial | lignicolous | coprophilous | etc.
    "ecology.microhabitat_notes",       # e.g. mossy ground, disturbed areas
]

TAXONOMIC_FIELDS = [
    "kingdom",
    "phylum",                           # e.g. Basidiomycetes — useful for broad filtering
    "order",
    "family",
    "genus",
    "species",
    "common_names",
    "synonyms",
]


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

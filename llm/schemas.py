"""
Pydantic schemas for LLM-structured outputs.

Three categories:
    1. Extraction — used at ingestion time to parse source text into features
    2. Reconciliation — used to merge multiple source observations
    3. Explanation — used at query time to generate human-readable summaries

These are LLM output contracts (Instructor validates against them).
API response schemas live in api/schemas.py.
"""

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Shared coercion helpers
# ---------------------------------------------------------------------------

_FALSY = {"false", "no", "none", "null", "0", "absent", ""}


def _coerce_bool(v) -> bool | None:
    """
    Coerce LLM boolean output to Python bool.
    - Already a bool → keep as-is
    - None → None
    - String like "false"/"no"/"none"/"absent" → False
    - Any other non-empty string (e.g. "slightly depressed", "yes", "present") → True
    - Numbers: 0 → False, non-zero → True
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() not in _FALSY
    return bool(v)


def _coerce_bool_fields(data: dict, fields: tuple) -> dict:
    """Apply _coerce_bool to a set of field names in a dict."""
    for field in fields:
        if field in data:
            data[field] = _coerce_bool(data[field])
    return data


# ---------------------------------------------------------------------------
# 1. Feature extraction schemas (ingestion time)
# ---------------------------------------------------------------------------


class ExtractedCapFeatures(BaseModel):
    shape: str | None = Field(
        None,
        description=(
            "Cap shape. Examples: convex, broadly convex, flat, depressed, "
            "umbonate, conical, bell-shaped, irregular"
        ),
    )
    colors: list[str] = Field(
        default_factory=list,
        description=(
            "Observed colors when fresh. "
            "Examples: ['scarlet red', 'orange', 'yellow-orange', 'bay brown']"
        ),
    )
    color_faded: str | None = Field(
        None,
        description="Color when dried or faded. Examples: buff, pale ochraceous, brownish",
    )
    color_pattern: str | None = Field(
        None,
        description=(
            "Pattern of color on cap. "
            "Examples: uniform, darker at center, two-toned, mottled, streaked"
        ),
    )
    surface_texture: str | None = Field(
        None,
        description=(
            "Surface feel/appearance. "
            "Examples: smooth, slimy/viscid, dry, velvety, fibrous, hairy-scaly, powdery, matt"
        ),
    )
    surface_moisture: str | None = Field(
        None,
        description=("Surface moisture. Examples: dry, viscid, glutinous, hygrophanous"),
    )
    scales_or_warts: str | None = Field(
        None,
        description=(
            "Scales or wart-like patches on cap. "
            "Examples: 'white universal veil remnants', 'absent', 'fibrous grey-brown scales'"
        ),
    )
    margin_type: str | None = Field(
        None,
        description=(
            "Shape/form of cap edge. "
            "Examples: inrolled, wavy, even, striate (lined), fringed, lobed"
        ),
    )
    margin_lined_at_maturity: bool | None = Field(
        None,
        description=(
            "True if the margin becomes striate (lined/grooved) with age, "
            "e.g. as in Laccaria species"
        ),
    )
    bruising_color: str | None = Field(
        None,
        description="Color change when cap is handled or bruised",
    )
    central_depression: bool | None = Field(
        None,
        description="True if cap is depressed (funnel-like) at the center at maturity",
    )
    diameter_min_cm: float | None = Field(None, description="Minimum typical cap diameter in cm")
    diameter_max_cm: float | None = Field(None, description="Maximum typical cap diameter in cm")

    @model_validator(mode="before")
    @classmethod
    def coerce_bools(cls, data: dict) -> dict:
        if isinstance(data, dict):
            _coerce_bool_fields(data, ("margin_lined_at_maturity", "central_depression"))
        return data


class ExtractedHymeniumFeatures(BaseModel):
    """The spore-bearing surface type — determines which sub-section applies."""

    type: str | None = Field(
        None,
        description=(
            "Hymenium (spore-bearing surface) type. "
            "One of: gills, pores, teeth, ridges, smooth. "
            "Use 'ridges' for forking blunt ridges / false gills (Cantharellus, Craterellus). "
            "Use 'gills' only for true blade-like gills separable from the cap flesh. "
            "Use 'teeth' for spine-like projections (Hydnum). "
            "Use 'smooth' when no distinct structure (puffballs, Craterellus interior)."
        ),
    )


class ExtractedGillFeatures(BaseModel):
    """Gill details — fill only when hymenium.type == 'gills'."""

    attachment: str | None = Field(
        None,
        description=(
            "How gills attach to stem. "
            "Examples: free, adnate, decurrent, sinuate, adnexed, notched"
        ),
    )
    spacing: str | None = Field(
        None,
        description="Gill spacing. Examples: crowded, close, subdistant, distant",
    )
    color: str | None = Field(None, description="Gill color at maturity")
    color_with_age: str | None = Field(
        None,
        description=(
            "Gill color change as mushroom ages. "
            "Examples: 'white to pink then brown', 'yellow becoming rusty'"
        ),
    )
    thickness: str | None = Field(
        None,
        description="Gill thickness. Examples: thin, thick — thick gills diagnostic e.g. Laccaria",
    )
    texture: str | None = Field(
        None,
        description="Gill texture. Examples: waxy (Hygrocybe), brittle (Russula), normal/soft",
    )
    edge_texture: str | None = Field(
        None,
        description="Gill edge character. Examples: smooth, serrate, eroded, fimbriate",
    )


class ExtractedPoreFeatures(BaseModel):
    """Pore details — fill only when hymenium.type == 'pores'."""

    color: str | None = Field(
        None,
        description="Pore surface color when fresh. Examples: white, yellow, red, grey",
    )
    color_with_age: str | None = Field(
        None,
        description="Pore color change with age. Examples: 'white to olive-brown', 'reddish'",
    )
    bruising_color: str | None = Field(
        None,
        description=(
            "Color when bruised or pressed. "
            "Examples: blue (bluing), slowly orangish-brown, none/unchanged"
        ),
    )
    density_per_mm: str | None = Field(
        None,
        description="Number of pores per mm. Examples: '1-2 per mm', '3-4 per mm'",
    )


class ExtractedTubeFeatures(BaseModel):
    """Tube details — fill only when hymenium.type == 'pores'."""

    depth_mm: str | None = Field(
        None,
        description="Depth of tubes in mm. Examples: '5-20mm', 'up to 25mm', 'short (3-8mm)'",
    )


class ExtractedStemFeatures(BaseModel):
    color: str | None = Field(None, description="Main stem color")
    color_with_age: str | None = Field(
        None,
        description=(
            "Color change as stem ages. Examples: 'white becoming brownish at base', 'yellowing'"
        ),
    )
    surface_texture: str | None = Field(
        None,
        description=(
            "Stem surface. "
            "Examples: smooth, reticulate (net-like), fibrous, hairy, scaly, powdery, grooved"
        ),
    )
    reticulation: str | None = Field(
        None,
        description=(
            "Reticulation (net pattern) extent — key for boletes. "
            "Examples: none, partial (upper third), full"
        ),
    )
    shape: str | None = Field(
        None,
        description=(
            "Stem shape. "
            "Examples: equal (same width top to bottom), club-shaped (clavate), "
            "tapered base, bulbous, swollen base"
        ),
    )
    attachment_position: str | None = Field(
        None,
        description=("Where stipe attaches to cap. Examples: central, eccentric, lateral, absent"),
    )
    consistency: str | None = Field(
        None,
        description="Internal texture. Examples: firm, fibrous, spongy, brittle, cartilaginous",
    )
    hollow_or_solid: str | None = Field(
        None,
        description="Internal structure. Examples: hollow, stuffed (with pith), solid",
    )
    base_color: str | None = Field(
        None,
        description=(
            "Color specifically at stem base. Examples: white, bluish, yellowish, staining red"
        ),
    )
    basal_mycelium_color: str | None = Field(
        None,
        description=(
            "Color of mycelium threads at stem base. "
            "Examples: white, lilac (diagnostic for Laccaria), yellow"
        ),
    )
    finger_stain_color: str | None = Field(
        None,
        description=(
            "Color left on fingers when stem is rubbed. "
            "Examples: yellow (Retiboletus ornatipes), none"
        ),
    )
    bruising_color: str | None = Field(
        None,
        description="Color change when stem is bruised or cut",
    )
    height_min_cm: float | None = Field(None, description="Minimum typical stem height in cm")
    height_max_cm: float | None = Field(None, description="Maximum typical stem height in cm")
    diameter_min_cm: float | None = Field(None, description="Minimum typical stem diameter in cm")
    diameter_max_cm: float | None = Field(None, description="Maximum typical stem diameter in cm")


class ExtractedVeilFeatures(BaseModel):
    present: bool | None = Field(None, description="Whether any veil or ring is present")
    type: str | None = Field(
        None,
        description=(
            "Veil type. "
            "Examples: partial (ring/annulus), universal (volva + ring), "
            "cortina (cobweb-like), absent"
        ),
    )
    cortina_present: bool | None = Field(
        None,
        description=(
            "True if a cortina (cobweb-like partial veil) is present — "
            "key Cortinarius differentiator"
        ),
    )
    shape: str | None = Field(
        None,
        description="Ring shape. Examples: pendant (skirt-like), ascending, flaring, fragile",
    )
    color: str | None = None
    ring_position: str | None = Field(
        None,
        description="Ring position on stem. Examples: superior, median, inferior, apical",
    )
    ring_mobility: str | None = Field(
        None,
        description="Whether ring slides. Examples: fixed, movable",
    )
    ring_persistence: str | None = Field(
        None,
        description="Ring durability. Examples: persistent, fugacious, ring_zone",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_bools(cls, data: dict) -> dict:
        if isinstance(data, dict):
            _coerce_bool_fields(data, ("present", "cortina_present"))
        return data


class ExtractedVolvaFeatures(BaseModel):
    present: bool | None = Field(None, description="Whether a volva (basal sac) is present")
    type: str | None = Field(
        None,
        description=(
            "Volva form. "
            "Examples: sac-like (cup-shaped, free margin), sheathing (close-fitting), "
            "friable (rings of scales)"
        ),
    )
    shape: str | None = Field(
        None,
        description=(
            "Volva shape. "
            "Examples: cup-shaped, bag-like (saccate), ridged collar, rings of scales around bulb"
        ),
    )
    color: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_bools(cls, data: dict) -> dict:
        if isinstance(data, dict):
            _coerce_bool_fields(data, ("present",))
        return data


class ExtractedFleshFeatures(BaseModel):
    color: str | None = Field(None, description="Flesh color when cut")
    bruising_color: str | None = Field(
        None,
        description=(
            "Color change when cut or bruised. "
            "Examples: blue (Gyroporus cyanescens), red, none/unchanged"
        ),
    )
    latex_presence: bool | None = Field(
        None,
        description="True if mushroom exudes latex (milk) when cut or broken. False if absent.",
    )
    latex_color: str | None = Field(
        None,
        description=(
            "Color of latex when present. "
            "Examples: white, white_to_yellow (changing), blue, red, orange, clear/watery"
        ),
    )
    odor: str | None = Field(
        None,
        description=(
            "Smell. Examples: not distinctive, pleasant mushroomy, anise/aniseed, fishy, "
            "foul/unpleasant, mealy/farinaceous, fruity, garlic-like"
        ),
    )
    taste: str | None = Field(
        None,
        description=(
            "Taste (where safely testable). "
            "Examples: mild, bitter, acrid/peppery (Russula), farinaceous/mealy, not distinctive"
        ),
    )
    texture: str | None = Field(
        None,
        description=(
            "Texture when handled/cut. "
            "Examples: firm, soft, brittle (snaps cleanly like Russula), insubstantial, watery"
        ),
    )
    hyphal_structure: str | None = Field(
        None,
        description="Cell type. Examples: homoiomerous, heteromerous",
    )
    cap_stem_consistency: str | None = Field(
        None,
        description=("Whether cap and stem flesh match. Examples: homogeneous, heterogeneous"),
    )
    quantity: str | None = Field(
        None,
        description=(
            "Relative amount of flesh. "
            "Examples: insubstantial/thin (e.g. small Mycena), moderate, thick"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_bools(cls, data: dict) -> dict:
        if isinstance(data, dict):
            _coerce_bool_fields(data, ("latex_presence",))
        return data


class ExtractedSporeFeatures(BaseModel):
    """Microscopic spore measurements and properties."""

    shape: str | None = Field(
        None,
        description=(
            "Spore shape. "
            "Examples: globose (round), ellipsoid, subfusoid (spindle-like), "
            "amygdaliform (almond-shaped), cylindrical"
        ),
    )
    length_min_um: float | None = Field(
        None, description="Minimum spore length in micrometres (µm)"
    )
    length_max_um: float | None = Field(
        None, description="Maximum spore length in micrometres (µm)"
    )
    width_min_um: float | None = Field(None, description="Minimum spore width in micrometres (µm)")
    width_max_um: float | None = Field(None, description="Maximum spore width in micrometres (µm)")
    ornamentation: str | None = Field(
        None,
        description=(
            "Spore surface ornamentation. "
            "Examples: smooth, echinulate (spiny), warty, reticulate (netted), striate"
        ),
    )
    spine_length_um: float | None = Field(
        None,
        description=(
            "Spine/echinus length in µm for echinulate spores. "
            "Diagnostic e.g. 1.5–3 µm in Laccaria"
        ),
    )
    spine_base_width_um: float | None = Field(
        None,
        description="Spine base width in µm — further diagnostic detail for Laccaria genus",
    )
    amyloidity: str | None = Field(
        None,
        description=(
            "Reaction with Melzer's reagent. "
            "Examples: amyloid (blue-black), inamyloid (no reaction), dextrinoid (reddish-brown)"
        ),
    )
    color_in_KOH: str | None = Field(
        None,
        description=(
            "Spore wall color when mounted in KOH solution. "
            "Examples: pale yellow, hyaline, brownish"
        ),
    )


class ExtractedMicroscopicFeatures(BaseModel):
    """Microscopic anatomical details beyond the spores."""

    basidia_spore_count: str | None = Field(
        None,
        description=(
            "Number of spores per basidium. Examples: 4-spored, 2-spored, mixed 2- and 4-spored"
        ),
    )
    cheilocystidia_shape: str | None = Field(
        None,
        description=(
            "Shape of cystidia on gill edge. "
            "Examples: narrowly cylindric, subclavate, capitate, fusoid, absent"
        ),
    )
    cheilocystidia_dims_um: str | None = Field(
        None,
        description=(
            "Dimensions of cheilocystidia. "
            "Format: 'length_min–length_max × width_min–width_max'. Example: '25–65 × 4–12'"
        ),
    )
    pleurocystidia_shape: str | None = Field(
        None,
        description=(
            "Shape of cystidia on gill faces. Examples: fusoid-ventricose, metuloid, absent"
        ),
    )
    pleurocystidia_dims_um: str | None = Field(
        None,
        description="Dimensions of pleurocystidia in µm. Example: '40–80 × 8–15'",
    )
    cystidia_color_in_KOH: str | None = Field(
        None,
        description="Color of cystidia contents in KOH. Examples: hyaline, pale yellow, brown",
    )
    pileipellis_type: str | None = Field(
        None,
        description=(
            "Structure of the cap cuticle. "
            "Examples: cutis (parallel hyphae), trichoderm (upright hyphae), "
            "ixocutis (gelatinised), hymeniderm"
        ),
    )
    pileipellis_element_width_um: str | None = Field(
        None,
        description="Width of pileipellis hyphae in µm. Example: '3–8 µm', '5–12 µm wide'",
    )
    pileipellis_terminal_cell_shape: str | None = Field(
        None,
        description=(
            "Shape of terminal (tip) cells of pileipellis. "
            "Examples: subclavate, capitate, rounded, cylindric"
        ),
    )


class ExtractedChemicalReactions(BaseModel):
    """Spot-test chemical reactions on cap surface and flesh."""

    KOH_cap: str | None = Field(
        None,
        description=(
            "KOH reaction on cap surface. "
            "Examples: yellow, orange, red, negative/no reaction, blackening"
        ),
    )
    KOH_flesh: str | None = Field(
        None,
        description="KOH reaction on cut flesh. Examples: yellow, orange, negative, turning dark",
    )
    NH4OH_cap: str | None = Field(
        None,
        description="Ammonium hydroxide reaction on cap. Examples: yellow, green, negative",
    )
    NH4OH_flesh: str | None = Field(
        None,
        description="Ammonium hydroxide reaction on flesh. Examples: yellow, negative",
    )
    FeSO4_cap: str | None = Field(
        None,
        description="Iron sulfate (FeSO₄) reaction on cap. Examples: blue-green, pink, negative",
    )
    FeSO4_flesh: str | None = Field(
        None,
        description=(
            "Iron sulfate (FeSO₄) reaction on flesh. Examples: blue-green, grey-green, negative"
        ),
    )


class ExtractedEcologicalFeatures(BaseModel):
    trophic_mode: str | None = Field(
        None,
        description="Nutritional strategy. Examples: mycorrhizal, saprotrophic, parasitic",
    )
    habitat_types: list[str] = Field(
        default_factory=list,
        description=(
            "Habitat types where found. "
            "Examples: ['hardwood forest', 'conifer forest', 'grassland', 'heathland']"
        ),
    )
    substrate: str | None = Field(
        None,
        description=(
            "What it grows on. "
            "Examples: soil, dead wood, living tree, dung, leaf litter, mossy ground"
        ),
    )
    associated_trees: list[str] = Field(
        default_factory=list,
        description=(
            "Tree partners (for mycorrhizal species). "
            "Examples: ['oak', 'beech', 'birch', 'pine', 'spruce']"
        ),
    )
    fruiting_seasons: list[str] = Field(
        default_factory=list,
        description=(
            "Season(s) when fruiting bodies appear. "
            "Examples: ['spring'], ['summer', 'fall'], ['winter']"
        ),
    )
    fruiting_months: str | None = Field(
        None,
        description=(
            "More specific fruiting period. "
            "Examples: 'July–September', 'late spring through summer', 'October–December'"
        ),
    )
    geographic_regions: list[str] = Field(
        default_factory=list,
        description=(
            "Geographic distribution. "
            "Examples: ['Europe', 'North America', 'East Asia', 'cosmopolitan']"
        ),
    )
    altitude_notes: str | None = Field(
        None,
        description=(
            "Altitude range or preference. "
            "Examples: 'lowland to montane', 'alpine zones', 'below 1000m'"
        ),
    )
    growth_position: str | None = Field(
        None,
        description=(
            "Where it grows relative to substrate. "
            "Examples: terrestrial (ground), lignicolous (on wood), coprophilous (on dung)"
        ),
    )
    microhabitat_notes: str | None = Field(
        None,
        description=(
            "Specific microhabitat detail. "
            "Examples: 'mossy ground near streams', 'disturbed soil', 'under bracken'"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_null_lists(cls, data: dict) -> dict:
        if isinstance(data, dict):
            for field in (
                "habitat_types",
                "associated_trees",
                "fruiting_seasons",
                "geographic_regions",
            ):
                if data.get(field) is None:
                    data[field] = []
        return data


# ---------------------------------------------------------------------------
# Grouped extraction schemas (2-pass pipeline)
# ---------------------------------------------------------------------------


class ExtractedVariety(BaseModel):
    name: str = Field(
        description="Variety/form name, e.g. 'var. alba', 'f. flavivolvata', 'subsp. muscaria'"
    )
    description: str | None = Field(
        None,
        description="Free-text summary of how this variety differs from the nominal form",
    )
    differing_features: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Key features that differ from the species baseline. "
            "Keys are descriptive names (e.g. 'cap_color', 'habitat'), "
            "values are the variety-specific value."
        ),
    )
    geographic_notes: str | None = Field(
        None,
        description="Geographic range of this variety if different from species",
    )
    edibility_note: str | None = Field(
        None,
        description="Only if edibility differs from the nominal species",
    )


class Pass1IdentityFeatures(BaseModel):
    """Pass 1 — identity, taxonomy, body plan, safety (~18 fields)."""

    scientific_name: str
    species_epithet: str | None = None
    common_names: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    kingdom: str | None = None
    phylum: str | None = None
    order: str | None = None
    family: str | None = None
    genus: str | None = None
    overall_body_form: str | None = None
    overall_size_class: str | None = None
    growth_habit: str | None = None
    hymenium_type: str | None = None
    edibility_status: str | None = None
    known_toxins: list[str] = Field(default_factory=list)
    known_lookalikes: list[str] = Field(default_factory=list)
    varieties: list[ExtractedVariety] = Field(
        default_factory=list,
        description=(
            "Named varieties, subspecies, or forms with features differing from the nominal taxon. "
            "Leave empty if the source text describes no distinct varieties."
        ),
    )
    extraction_notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_null_lists(cls, data: dict) -> dict:
        if isinstance(data, dict):
            for field in (
                "common_names",
                "synonyms",
                "known_toxins",
                "known_lookalikes",
                "varieties",
            ):
                if data.get(field) is None:
                    data[field] = []
        return data


class Pass2CapFeatures(BaseModel):
    """Pass 2 Group A — cap & surface (13 fields)."""

    cap: ExtractedCapFeatures = Field(default_factory=ExtractedCapFeatures)


class Pass2HymeniumFeatures(BaseModel):
    """Pass 2 Group B — hymenium details (13 fields)."""

    gills: ExtractedGillFeatures = Field(default_factory=ExtractedGillFeatures)
    pores: ExtractedPoreFeatures = Field(default_factory=ExtractedPoreFeatures)
    tubes: ExtractedTubeFeatures = Field(default_factory=ExtractedTubeFeatures)
    spore_print_color: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_null_nested(cls, data: dict) -> dict:
        if isinstance(data, dict):
            for field in ("gills", "pores", "tubes"):
                if data.get(field) is None:
                    data[field] = {}
        return data


class Pass2StemVeilFeatures(BaseModel):
    """Pass 2 Group C — stem, veil & volva (26 fields)."""

    stem: ExtractedStemFeatures = Field(default_factory=ExtractedStemFeatures)
    veil: ExtractedVeilFeatures = Field(default_factory=ExtractedVeilFeatures)
    volva: ExtractedVolvaFeatures = Field(default_factory=ExtractedVolvaFeatures)

    @model_validator(mode="before")
    @classmethod
    def coerce_null_nested(cls, data: dict) -> dict:
        if isinstance(data, dict):
            for field in ("stem", "veil", "volva"):
                if data.get(field) is None:
                    data[field] = {}
        return data


class Pass2FleshChemFeatures(BaseModel):
    """Pass 2 Group D — flesh & chemistry (15 fields)."""

    flesh: ExtractedFleshFeatures = Field(default_factory=ExtractedFleshFeatures)
    chemical: ExtractedChemicalReactions = Field(default_factory=ExtractedChemicalReactions)

    @model_validator(mode="before")
    @classmethod
    def coerce_null_nested(cls, data: dict) -> dict:
        if isinstance(data, dict):
            for field in ("flesh", "chemical"):
                if data.get(field) is None:
                    data[field] = {}
        return data


class Pass2SporeEcoFeatures(BaseModel):
    """Pass 2 Group E — spores, microscopic & ecology (29 fields)."""

    spore: ExtractedSporeFeatures = Field(default_factory=ExtractedSporeFeatures)
    microscopic: ExtractedMicroscopicFeatures = Field(default_factory=ExtractedMicroscopicFeatures)
    ecology: ExtractedEcologicalFeatures = Field(default_factory=ExtractedEcologicalFeatures)

    @model_validator(mode="before")
    @classmethod
    def coerce_null_nested(cls, data: dict) -> dict:
        if isinstance(data, dict):
            for field in ("spore", "microscopic", "ecology"):
                if data.get(field) is None:
                    data[field] = {}
        return data


class ExtractedSpeciesFeatures(BaseModel):
    """
    Full feature extraction for one species from one source.

    This is the schema Instructor validates against when parsing source text.
    One instance per species per source → stored as Layer 1 (source_observations).
    """

    # Identity / taxonomy
    scientific_name: str = Field(description="Full binomial name, e.g. 'Amanita caesarea'")
    species_epithet: str | None = Field(
        None,
        description=(
            "The species part of the binomial only. Example: for Amanita caesarea → 'caesarea'"
        ),
    )
    common_names: list[str] = Field(
        default_factory=list,
        description=(
            "Common names in any language. Examples: [\"Caesar's mushroom\", 'Fly agaric', 'Cep']"
        ),
    )
    synonyms: list[str] = Field(
        default_factory=list,
        description=(
            "Taxonomic synonyms (older valid names). "
            "Example: ['Boletus edulis Bull.', 'Agaricus esculentus Wulfen']"
        ),
    )
    kingdom: str | None = Field(None, description="Kingdom. Almost always: Fungi")
    phylum: str | None = Field(
        None,
        description="Phylum. Examples: Basidiomycota, Ascomycota",
    )
    order: str | None = Field(
        None,
        description=(
            "Taxonomic order. Examples: Agaricales, Boletales, Russulales, Cantharellales"
        ),
    )
    family: str | None = Field(
        None,
        description=(
            "Taxonomic family. Examples: Amanitaceae, Boletaceae, Russulaceae, Cantharellaceae"
        ),
    )
    genus: str | None = Field(
        None,
        description=("Taxonomic genus. Examples: Amanita, Boletus, Russula, Cantharellus"),
    )

    # Morphological
    cap: ExtractedCapFeatures = Field(default_factory=ExtractedCapFeatures)
    hymenium: ExtractedHymeniumFeatures = Field(default_factory=ExtractedHymeniumFeatures)
    gills: ExtractedGillFeatures = Field(default_factory=ExtractedGillFeatures)
    pores: ExtractedPoreFeatures = Field(default_factory=ExtractedPoreFeatures)
    tubes: ExtractedTubeFeatures = Field(default_factory=ExtractedTubeFeatures)
    stem: ExtractedStemFeatures = Field(default_factory=ExtractedStemFeatures)
    veil: ExtractedVeilFeatures = Field(default_factory=ExtractedVeilFeatures)
    volva: ExtractedVolvaFeatures = Field(default_factory=ExtractedVolvaFeatures)
    flesh: ExtractedFleshFeatures = Field(default_factory=ExtractedFleshFeatures)
    spore_print_color: str | None = Field(
        None,
        description=(
            "Color of spore deposit. "
            "Examples: white, pink, brown, purple-brown, black, rusty-brown, olive"
        ),
    )
    spore: ExtractedSporeFeatures = Field(default_factory=ExtractedSporeFeatures)
    microscopic: ExtractedMicroscopicFeatures = Field(default_factory=ExtractedMicroscopicFeatures)
    chemical: ExtractedChemicalReactions = Field(default_factory=ExtractedChemicalReactions)
    overall_size_class: str | None = Field(
        None,
        description="Relative size. One of: small, medium, large",
    )
    overall_body_form: str | None = Field(
        None,
        description=("Body plan. Examples: agaricoid, boletoid, gasteroid, tremelloid"),
    )
    growth_habit: str | None = Field(
        None,
        description=("Growth arrangement. Examples: solitary, scattered, gregarious, caespitose"),
    )

    # Ecological
    ecology: ExtractedEcologicalFeatures = Field(default_factory=ExtractedEcologicalFeatures)

    # Safety
    edibility_status: str | None = Field(
        None,
        description=(
            "Edibility. "
            "One of: edible, choice (prized edible), conditionally edible, inedible, toxic, deadly"
        ),
    )
    known_toxins: list[str] = Field(
        default_factory=list,
        description=(
            "Named toxins if known. "
            "Examples: ['amatoxins', 'ibotenic acid', 'muscimol', 'gyromitrin']"
        ),
    )
    known_lookalikes: list[str] = Field(
        default_factory=list,
        description="Species mentioned as lookalikes in the source text. Use scientific names.",
    )
    varieties: list[ExtractedVariety] = Field(
        default_factory=list,
        description=(
            "Named varieties, subspecies, or forms with features differing from the nominal taxon. "
            "Leave empty if the source text describes no distinct varieties."
        ),
    )

    extraction_notes: str | None = Field(
        None,
        description=(
            "Record any ambiguity, uncertainty, or conflicting information "
            "found in the source text."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_nulls(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        # Coerce null list fields to []
        for field in (
            "common_names",
            "known_toxins",
            "known_lookalikes",
            "synonyms",
            "varieties",
        ):
            if data.get(field) is None:
                data[field] = []
        # Coerce null nested objects to {} so sub-models get their defaults
        # (LLMs often return null for sections that don't apply, e.g. pores on a gill mushroom)
        for field in (
            "cap",
            "hymenium",
            "gills",
            "pores",
            "tubes",
            "stem",
            "veil",
            "volva",
            "flesh",
            "spore",
            "microscopic",
            "chemical",
            "ecology",
        ):
            if data.get(field) is None:
                data[field] = {}
        return data


# ---------------------------------------------------------------------------
# 2. Reconciliation schemas (merging multiple sources)
# ---------------------------------------------------------------------------


class ReconciliationResult(BaseModel):
    """
    LLM output when reconciling multiple source observations for one species.
    """

    reconciled_features: ExtractedSpeciesFeatures = Field(
        description="The canonical merged feature profile for this species"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in the reconciled profile (0-1)",
    )
    conflicts: list[str] = Field(
        default_factory=list,
        description="Fields where sources disagreed and reconciliation was uncertain",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_null_lists(cls, data: dict) -> dict:
        if isinstance(data, dict) and data.get("conflicts") is None:
            data["conflicts"] = []
        return data

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
        description=(
            "2-3 sentence 'at a glance' summary of the most notable "
            "lookalikes and the key reason they cause confusion."
        )
    )
    notable_pairs: list[str] = Field(
        description=(
            "The 2-3 most important lookalike relationships to highlight, "
            "each as a concise sentence explaining what makes them confusable "
            "and what the critical distinguishing feature is."
        )
    )
    safety_warning: str = Field(
        description=(
            "Safety note if any lookalikes are toxic or deadly. Must be prominent and unambiguous."
        )
    )

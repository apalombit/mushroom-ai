"""Unit tests for ingestion/extract.py — prompt generation and grouped extraction."""

import yaml
from pydantic import BaseModel

from ingestion.extract import (
    _GROUP_A_VOCAB_KEYS,
    _GROUP_B_VOCAB_KEYS,
    _GROUP_C_VOCAB_KEYS,
    _GROUP_D_VOCAB_KEYS,
    _GROUP_E_VOCAB_KEYS,
    _PASS1_VOCAB_KEYS,
    _SKIP_ALIASES_FOR,
    _VOCAB_PATH,
    _VOCAB_TO_SECTION,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_GROUP_A,
    SYSTEM_PROMPT_GROUP_B,
    SYSTEM_PROMPT_GROUP_C,
    SYSTEM_PROMPT_GROUP_D,
    SYSTEM_PROMPT_GROUP_E,
    SYSTEM_PROMPT_PASS1,
    _build_vocab_guidance,
    merge_extraction_results,
)
from llm.schemas import (
    ExtractedCapFeatures,
    ExtractedEcologicalFeatures,
    ExtractedFleshFeatures,
    ExtractedGillFeatures,
    ExtractedMicroscopicFeatures,
    ExtractedPoreFeatures,
    ExtractedSpeciesFeatures,
    ExtractedSporeFeatures,
    ExtractedStemFeatures,
    ExtractedTubeFeatures,
    ExtractedVeilFeatures,
    ExtractedVolvaFeatures,
    Pass1IdentityFeatures,
    Pass2CapFeatures,
    Pass2FleshChemFeatures,
    Pass2HymeniumFeatures,
    Pass2SporeEcoFeatures,
    Pass2StemVeilFeatures,
)

# ---------------------------------------------------------------------------
# Existing tests (monolithic prompt)
# ---------------------------------------------------------------------------


def test_system_prompt_includes_vocab_canonical_values():
    """Every YAML feature should have at least some canonical values in the prompt."""
    with open(_VOCAB_PATH) as f:
        vocab = yaml.safe_load(f)

    for yaml_key in _VOCAB_TO_SECTION:
        if yaml_key not in vocab:
            continue
        values = list(vocab[yaml_key]["canonical_values"].keys())
        # At least 2 canonical values should appear in the prompt
        found = [v for v in values if v in SYSTEM_PROMPT]
        assert len(found) >= 2, (
            f"Feature '{yaml_key}': expected ≥2 canonical values in prompt, "
            f"found {len(found)}: {found}"
        )


def test_system_prompt_no_stale_boolean_hygrophanous():
    """The old standalone 'hygrophanous: true if ...' boolean field should be gone."""
    assert "hygrophanous: true if" not in SYSTEM_PROMPT


def test_system_prompt_no_stale_growth_pattern():
    """The old 'growth_pattern' field should be replaced by 'growth_habit'."""
    assert "growth_pattern" not in SYSTEM_PROMPT
    assert "growth_habit" in SYSTEM_PROMPT


def test_system_prompt_has_new_fields():
    """All 14 new Stage-2 fields should appear in the prompt."""
    new_fields = [
        "growth_habit",
        "overall_body_form",
        "surface_moisture",
        "latex",
        "bruising_color",
        "edge_texture",
        "attachment_position",
        "ring_position",
        "ring_mobility",
        "ring_persistence",
        "hyphal_structure",
        "cap_stem_consistency",
    ]
    for field in new_fields:
        assert field in SYSTEM_PROMPT, f"New field '{field}' missing from SYSTEM_PROMPT"


def test_system_prompt_preserves_contextual_notes():
    """Hardcoded contextual notes should still be present."""
    assert 'Use "ridges" for species with forking' in SYSTEM_PROMPT
    assert "edibility_status: MUST be one of" in SYSTEM_PROMPT
    assert "only populate gills fields when hymenium.type" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Field coverage & overlap tests
# ---------------------------------------------------------------------------


def _leaf_field_paths(model: type[BaseModel], prefix: str = "") -> set[str]:
    """Recursively collect all leaf field paths from a Pydantic model."""
    paths = set()
    for name, field_info in model.model_fields.items():
        full = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
        annotation = field_info.annotation
        # Unwrap Optional (X | None)
        origin = getattr(annotation, "__origin__", None)
        if origin is type(int | str):  # types.UnionType
            args = [a for a in annotation.__args__ if a is not type(None)]
            annotation = args[0] if len(args) == 1 else annotation
        # Check if it's a nested BaseModel
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            paths |= _leaf_field_paths(annotation, full)
        else:
            paths.add(full)
    return paths


# Map each group model → the field paths it covers on ExtractedSpeciesFeatures.
# Pass1 fields are flat on ExtractedSpeciesFeatures, but hymenium_type → hymenium.type.
_PASS1_FIELD_PATHS = {
    "scientific_name",
    "species_epithet",
    "common_names",
    "synonyms",
    "kingdom",
    "phylum",
    "order",
    "family",
    "genus",
    "overall_body_form",
    "overall_size_class",
    "growth_habit",
    "hymenium.type",
    "edibility_status",
    "known_toxins",
    "known_lookalikes",
    "extraction_notes",
}

# Pass2 groups map directly: group field paths with their prefix match ESF paths.
_GROUP_A_FIELD_PATHS = _leaf_field_paths(ExtractedCapFeatures, "cap")
_GROUP_B_FIELD_PATHS = (
    _leaf_field_paths(ExtractedGillFeatures, "gills")
    | _leaf_field_paths(ExtractedPoreFeatures, "pores")
    | _leaf_field_paths(ExtractedTubeFeatures, "tubes")
    | {"spore_print_color"}
)
_GROUP_C_FIELD_PATHS = (
    _leaf_field_paths(ExtractedStemFeatures, "stem")
    | _leaf_field_paths(ExtractedVeilFeatures, "veil")
    | _leaf_field_paths(ExtractedVolvaFeatures, "volva")
)
_GROUP_D_FIELD_PATHS = _leaf_field_paths(ExtractedFleshFeatures, "flesh") | {
    "chemical.KOH_cap",
    "chemical.KOH_flesh",
    "chemical.NH4OH_cap",
    "chemical.NH4OH_flesh",
    "chemical.FeSO4_cap",
    "chemical.FeSO4_flesh",
}
_GROUP_E_FIELD_PATHS = (
    _leaf_field_paths(ExtractedSporeFeatures, "spore")
    | _leaf_field_paths(ExtractedMicroscopicFeatures, "microscopic")
    | _leaf_field_paths(ExtractedEcologicalFeatures, "ecology")
)


def test_all_fields_covered_across_groups():
    """Union of all 6 group field paths must cover every ExtractedSpeciesFeatures field."""
    esf_fields = _leaf_field_paths(ExtractedSpeciesFeatures)
    group_union = (
        _PASS1_FIELD_PATHS
        | _GROUP_A_FIELD_PATHS
        | _GROUP_B_FIELD_PATHS
        | _GROUP_C_FIELD_PATHS
        | _GROUP_D_FIELD_PATHS
        | _GROUP_E_FIELD_PATHS
    )
    missing = esf_fields - group_union
    assert not missing, f"Fields not covered by any group: {sorted(missing)}"


def test_no_field_overlap_between_groups():
    """No field path should appear in more than one group."""
    groups = [
        ("pass1", _PASS1_FIELD_PATHS),
        ("group_a", _GROUP_A_FIELD_PATHS),
        ("group_b", _GROUP_B_FIELD_PATHS),
        ("group_c", _GROUP_C_FIELD_PATHS),
        ("group_d", _GROUP_D_FIELD_PATHS),
        ("group_e", _GROUP_E_FIELD_PATHS),
    ]
    seen: dict[str, str] = {}
    overlaps = []
    for name, fields in groups:
        for f in fields:
            if f in seen:
                overlaps.append(f"{f} in both {seen[f]} and {name}")
            else:
                seen[f] = name
    assert not overlaps, f"Overlapping fields: {overlaps}"


# ---------------------------------------------------------------------------
# Merge correctness test
# ---------------------------------------------------------------------------


def test_merge_correctness():
    """Construct mock partial results, merge, verify field placement."""
    pass1 = Pass1IdentityFeatures(
        scientific_name="Amanita muscaria",
        species_epithet="muscaria",
        common_names=["fly agaric"],
        kingdom="Fungi",
        phylum="Basidiomycota",
        order="Agaricales",
        family="Amanitaceae",
        genus="Amanita",
        overall_body_form="agaricoid",
        overall_size_class="large",
        growth_habit="solitary",
        hymenium_type="gills",
        edibility_status="toxic",
        known_toxins=["ibotenic acid", "muscimol"],
        known_lookalikes=["Amanita caesarea"],
        extraction_notes="Well-described species",
    )
    cap = Pass2CapFeatures(
        cap=ExtractedCapFeatures(shape="convex", colors=["scarlet red", "orange"])
    )
    hymenium = Pass2HymeniumFeatures(
        gills=ExtractedGillFeatures(attachment="free", spacing="close"),
        spore_print_color="white",
    )
    stem_veil = Pass2StemVeilFeatures(
        stem=ExtractedStemFeatures(color="white", shape="equal"),
        veil=ExtractedVeilFeatures(present=True, type="partial"),
        volva=ExtractedVolvaFeatures(present=True, type="sac-like"),
    )
    flesh_chem = Pass2FleshChemFeatures(
        flesh=ExtractedFleshFeatures(color="white", odor="not distinctive"),
    )
    spore_eco = Pass2SporeEcoFeatures(
        spore=ExtractedSporeFeatures(shape="ellipsoid"),
        ecology=ExtractedEcologicalFeatures(trophic_mode="mycorrhizal"),
    )

    result = merge_extraction_results(pass1, cap, hymenium, stem_veil, flesh_chem, spore_eco)

    # Check type
    assert isinstance(result, ExtractedSpeciesFeatures)

    # Pass 1 fields
    assert result.scientific_name == "Amanita muscaria"
    assert result.genus == "Amanita"
    assert result.overall_body_form == "agaricoid"
    assert result.hymenium.type == "gills"
    assert result.edibility_status == "toxic"
    assert result.known_toxins == ["ibotenic acid", "muscimol"]
    assert result.growth_habit == "solitary"

    # Group A — cap
    assert result.cap.shape == "convex"
    assert result.cap.colors == ["scarlet red", "orange"]

    # Group B — hymenium details
    assert result.gills.attachment == "free"
    assert result.gills.spacing == "close"
    assert result.spore_print_color == "white"

    # Group C — stem, veil, volva
    assert result.stem.color == "white"
    assert result.veil.present is True
    assert result.volva.type == "sac-like"

    # Group D — flesh, chemical
    assert result.flesh.color == "white"
    assert result.flesh.odor == "not distinctive"

    # Group E — spore, microscopic, ecology
    assert result.spore.shape == "ellipsoid"
    assert result.ecology.trophic_mode == "mycorrhizal"


# ---------------------------------------------------------------------------
# Per-group prompt tests
# ---------------------------------------------------------------------------


def test_per_group_prompt_has_relevant_vocab():
    """Each group prompt must contain canonical values from its vocab key set."""
    with open(_VOCAB_PATH) as f:
        vocab = yaml.safe_load(f)

    pairs = [
        (_PASS1_VOCAB_KEYS, SYSTEM_PROMPT_PASS1, "pass1"),
        (_GROUP_A_VOCAB_KEYS, SYSTEM_PROMPT_GROUP_A, "group_a"),
        (_GROUP_B_VOCAB_KEYS, SYSTEM_PROMPT_GROUP_B, "group_b"),
        (_GROUP_C_VOCAB_KEYS, SYSTEM_PROMPT_GROUP_C, "group_c"),
        (_GROUP_D_VOCAB_KEYS, SYSTEM_PROMPT_GROUP_D, "group_d"),
        (_GROUP_E_VOCAB_KEYS, SYSTEM_PROMPT_GROUP_E, "group_e"),
    ]
    for keys, prompt, label in pairs:
        for key in keys:
            if key not in vocab:
                continue
            values = list(vocab[key]["canonical_values"].keys())
            found = [v for v in values if v in prompt]
            assert len(found) >= 2, (
                f"{label} prompt missing vocab for '{key}': "
                f"expected ≥2 values, found {len(found)}: {found}"
            )


def test_per_group_prompt_excludes_irrelevant_vocab():
    """Group prompts should NOT contain vocab section headers from other groups."""
    # _build_vocab_guidance outputs section headers like "\nSTEM\n- shape: ..."
    # Group A (cap) should not have a STEM section; Group C (stem) should not have CAP.
    # Check via section headers which are unambiguous, unlike field names.
    assert "\nSTEM\n" not in SYSTEM_PROMPT_GROUP_A, "Group A should not have STEM section"
    assert "\nGILLS\n" not in SYSTEM_PROMPT_GROUP_A, "Group A should not have GILLS section"
    assert "\nCAP\n" not in SYSTEM_PROMPT_GROUP_C, "Group C should not have CAP section"
    assert "\nFLESH\n" not in SYSTEM_PROMPT_GROUP_A, "Group A should not have FLESH section"
    assert "\nSPORES\n" not in SYSTEM_PROMPT_GROUP_D, "Group D should not have SPORES section"


def test_per_group_prompt_has_language_rules():
    """All group prompts must contain the LANGUAGE translation rules."""
    prompts = [
        SYSTEM_PROMPT_PASS1,
        SYSTEM_PROMPT_GROUP_A,
        SYSTEM_PROMPT_GROUP_B,
        SYSTEM_PROMPT_GROUP_C,
        SYSTEM_PROMPT_GROUP_D,
        SYSTEM_PROMPT_GROUP_E,
    ]
    for i, prompt in enumerate(prompts):
        assert "ALL values MUST be in English" in prompt, f"Prompt {i} missing language rules"
        assert "giallo" in prompt, f"Prompt {i} missing Italian color translations"
        assert "Faggio" in prompt, f"Prompt {i} missing Italian tree translations"


def test_per_group_prompt_has_header():
    """All group prompts must start with the mycologist header."""
    prompts = [
        SYSTEM_PROMPT_PASS1,
        SYSTEM_PROMPT_GROUP_A,
        SYSTEM_PROMPT_GROUP_B,
        SYSTEM_PROMPT_GROUP_C,
        SYSTEM_PROMPT_GROUP_D,
        SYSTEM_PROMPT_GROUP_E,
    ]
    for i, prompt in enumerate(prompts):
        assert prompt.startswith("You are a mycologist"), f"Prompt {i} missing mycologist header"
        assert "Extract ONLY what is explicitly stated" in prompt, (
            f"Prompt {i} missing general rules"
        )


# ---------------------------------------------------------------------------
# Alias inclusion tests
# ---------------------------------------------------------------------------


def test_aliases_appear_for_non_color_features():
    """Aliases should appear in `term (= alias1, alias2)` format for non-color features."""
    guidance = _build_vocab_guidance({"hymenium_type"})
    # hymenium_type has aliases: lamellae → gills, false gills → ridges, etc.
    assert "(= lamellae)" in guidance, "gills alias 'lamellae' missing"
    assert "ridges (= " in guidance, "ridges aliases missing"
    assert "gleba (= " in guidance, "gleba aliases missing"


def test_aliases_skipped_for_color_features():
    """color_palette and bruising_color should NOT have aliases in the guidance."""
    with open(_VOCAB_PATH) as f:
        vocab = yaml.safe_load(f)

    for key in _SKIP_ALIASES_FOR:
        if key not in vocab:
            continue
        aliases = vocab[key].get("aliases", {})
        if not aliases:
            continue
        guidance = _build_vocab_guidance({key})
        for alias in aliases:
            assert f"(= {alias})" not in guidance, f"Alias '{alias}' should be skipped for '{key}'"


def test_alias_format_matches_expected_pattern():
    """Verify the `term (= alias1, alias2)` output format."""
    guidance = _build_vocab_guidance({"hymenium_type"})
    # "gills (= lamellae)" — single alias
    assert "gills (= lamellae)" in guidance
    # "teeth" has multiple aliases: spines, spikes
    assert "teeth (= " in guidance
    # Terms without aliases should appear bare (no parentheses)
    assert "smooth" in guidance
    assert "smooth (=" not in guidance

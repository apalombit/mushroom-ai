"""
LLM-based feature extraction from source text.

Takes raw text about a mushroom species and extracts structured features
using Instructor-validated LLM calls.

Pipeline:
    Source text → LLM (with ExtractedSpeciesFeatures schema) → validated features
    → store as Layer 1 (SourceObservation) in Postgres
"""

import hashlib
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from config import settings
from db.connection import get_session
from db.models import SourceObservation
from llm.client import structured_completion
from llm.schemas import (
    ExtractedHymeniumFeatures,
    ExtractedSpeciesFeatures,
    Pass1IdentityFeatures,
    Pass2CapFeatures,
    Pass2FleshChemFeatures,
    Pass2HymeniumFeatures,
    Pass2SporeEcoFeatures,
    Pass2StemVeilFeatures,
)

logger = logging.getLogger(__name__)

# Truncate source text to keep prompts within a safe token budget.
# gemma3:27b has 8192 max output tokens; 8000 chars ≈ ~2000 tokens of input, leaving
# plenty of room for the structured output.
_MAX_TEXT_CHARS = 8_000

_VOCAB_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "reference" / "morphological_vocabulary.yaml"
)

# Map YAML feature keys → (section_header, field_name) for prompt generation.
_VOCAB_TO_SECTION = {
    "hymenium_type": ("HYMENIUM", "type"),
    "overall_body_form": ("OVERALL", "overall_body_form"),
    "cap_shape": ("CAP", "shape"),
    "cap_surface_moisture": ("CAP", "surface_moisture"),
    "surface_texture": ("CAP / STEM", "surface_texture"),
    "cap_margin": ("CAP", "margin_type"),
    "bruising_color": ("BRUISING", "bruising_color"),
    "gill_attachment": ("GILLS", "attachment"),
    "gill_spacing": ("GILLS", "spacing"),
    "gill_edge_texture": ("GILLS", "edge_texture"),
    "stem_shape": ("STEM", "shape"),
    "stem_interior": ("STEM", "hollow_or_solid"),
    "stipe_attachment_position": ("STEM", "attachment_position"),
    "veil_type": ("VEIL", "type"),
    "ring_shape": ("VEIL", "ring_shape"),
    "ring_position": ("VEIL", "ring_position"),
    "ring_mobility": ("VEIL", "ring_mobility"),
    "ring_persistence": ("VEIL", "ring_persistence"),
    "volva_type": ("VOLVA", "volva_type"),
    "flesh_odor": ("FLESH", "odor"),
    "flesh_taste": ("FLESH", "taste"),
    "flesh_texture": ("FLESH", "texture"),
    "flesh_hyphal_structure": ("FLESH", "hyphal_structure"),
    "flesh_cap_stem_consistency": ("FLESH", "cap_stem_consistency"),
    "latex": ("FLESH", "latex"),
    "spore_print_color": ("SPORE PRINT", "spore_print_color"),
    "spore_shape": ("SPORES", "shape"),
    "spore_ornamentation": ("SPORES", "ornamentation"),
    "spore_amyloidity": ("SPORES", "amyloidity"),
    "color_palette": ("COLORS", "all color fields"),
    "growth_habit": ("ECOLOGY", "growth_habit"),
    "pileipellis_type": ("MICROSCOPIC", "pileipellis_type"),
}


# Features with too many aliases (color-heavy) — already covered by LANGUAGE rules.
_SKIP_ALIASES_FOR = {"color_palette", "bruising_color"}


def _build_vocab_guidance(feature_keys: set[str] | None = None) -> str:
    """Build the canonical-values prompt section from the vocabulary YAML.

    Args:
        feature_keys: If provided, only include vocab entries whose YAML key
            is in this set. None means include all (monolithic prompt).
    """
    with open(_VOCAB_PATH) as f:
        vocab = yaml.safe_load(f)

    # Group features by section header
    sections: dict[str, list[str]] = defaultdict(list)
    for yaml_key, (section, field) in _VOCAB_TO_SECTION.items():
        if feature_keys is not None and yaml_key not in feature_keys:
            continue
        if yaml_key not in vocab:
            continue
        canonical = vocab[yaml_key]["canonical_values"]

        # Build reverse map: canonical_term → [alias1, alias2, ...]
        alias_map: dict[str, list[str]] = defaultdict(list)
        if yaml_key not in _SKIP_ALIASES_FOR:
            for alias, canon in vocab[yaml_key].get("aliases", {}).items():
                alias_map[canon].append(alias)

        # Format each term, appending aliases if present
        parts = []
        for term in canonical:
            if term in alias_map:
                aka = ", ".join(alias_map[term])
                parts.append(f"{term} (= {aka})")
            else:
                parts.append(term)
        line = f"- {field}: {' | '.join(parts)}"
        sections[section].append(line)

    # Build the output block
    lines = []
    for section, entries in sections.items():
        lines.append(f"\n{section}")
        lines.extend(entries)

    return "\n".join(lines)


# ── Per-group vocabulary key sets ─────────────────────────────────────────────

_PASS1_VOCAB_KEYS = {"hymenium_type", "overall_body_form", "growth_habit"}

_GROUP_A_VOCAB_KEYS = {
    "cap_shape",
    "cap_surface_moisture",
    "surface_texture",
    "cap_margin",
    "bruising_color",
    "color_palette",
}
_GROUP_B_VOCAB_KEYS = {
    "hymenium_type",
    "gill_attachment",
    "gill_spacing",
    "gill_edge_texture",
    "bruising_color",
    "spore_print_color",
    "color_palette",
}
_GROUP_C_VOCAB_KEYS = {
    "stem_shape",
    "stem_interior",
    "stipe_attachment_position",
    "surface_texture",
    "veil_type",
    "ring_shape",
    "ring_position",
    "ring_mobility",
    "ring_persistence",
    "volva_type",
    "bruising_color",
    "color_palette",
}
_GROUP_D_VOCAB_KEYS = {
    "flesh_odor",
    "flesh_taste",
    "flesh_texture",
    "flesh_hyphal_structure",
    "flesh_cap_stem_consistency",
    "latex",
    "bruising_color",
    "color_palette",
}
_GROUP_E_VOCAB_KEYS = {
    "spore_shape",
    "spore_ornamentation",
    "spore_amyloidity",
    "pileipellis_type",
    "growth_habit",
}


# ── System prompt: header (hardcoded general + language rules) ──────────────

_SYSTEM_PROMPT_HEADER = """\
You are a mycologist extracting structured features from species descriptions.

GENERAL RULES
- Extract ONLY what is explicitly stated in the source text. Never guess or infer.
- Use null for any feature not mentioned — omission is better than a wrong value.
- Record ambiguity or conflicting information in extraction_notes.
- scientific_name must match the queried species name exactly.
- For list fields, include every value mentioned in the text.

LANGUAGE
- ALL values MUST be in English — translate any Italian, French, German, or other
  foreign-language terms before writing them.
- Color examples: giallo → yellow, rosso → red, bruno → brown, nero → black,
  bianco/biancastro/biancastra → white/whitish, verde → green, arancione → orange,
  vinoso → wine-colored, lilacino/lilla → lilac, ocra → ochre, grigio → grey,
  rosa → pink, viola → violet, beige → beige, crema → cream.
- Compound color forms must also be translated: "vinoso-bruno" → "wine-brown",
  "rosa-lilacino" → "rose-lilac", "giallo-brunastro" → "yellowish-brown",
  "rosso-arancio" → "red-orange", "giallo-ocra" → "yellow-ochre".
- Non-color Italian words to translate: nullo/assente → none/absent,
  dolciastro → sweetish, soda/sodo → firm, acre/acro → acrid,
  farinoso → mealy/farinaceous, fruttato → fruity, mite → mild.
- Tree names to translate: Abete rosso → Norway spruce, Abete bianco → silver fir,
  Faggio → beech, Castagno → chestnut, Quercia → oak, Pino → pine,
  Betulla → birch, Leccio → holm oak, Carpino → hornbeam.
- Do NOT copy foreign words verbatim, even partially (no "biancastra", "dolciastro",
  "Abete rosso", "Faggio")."""

# ── System prompt: contextual notes beyond simple value lists ───────────────

_SYSTEM_PROMPT_NOTES = """

FIELD-SPECIFIC NOTES

HYMENIUM
- Use "ridges" for species with forking/blunt ridges or "false gills" that run down
  the stem and fork repeatedly (e.g. Cantharellus, Craterellus). These are NOT true
  gills — they are blunt, vein-like, and cannot be separated from the cap flesh.
- Use "gills" only for true blade-like gills that can be cleanly separated.
- Use "smooth" for species with no distinct hymenophore structure (e.g. Craterellus
  cornucopioides interior, puffballs).

GILLS — only populate gills fields when hymenium.type == "gills"
- color_with_age: e.g. "white to pink then brown", "yellow becoming rusty"
- thickness: thin | thick (thick gills are diagnostic e.g. for Laccaria)
- texture: waxy (Hygrocybe), brittle (Russula/Lactarius), normal/soft

PORES — only populate pores fields when hymenium.type == "pores"
- color: pore surface color when fresh
- bruising_color: "blue" (bluing), "slowly orangish-brown", "none"
- density_per_mm: e.g. "1-2 per mm", "3-4 per mm"

CAP
- colors: list of colors IN ENGLISH when fresh (translate from source language)
- color_faded: color when dry/faded IN ENGLISH
- color_pattern: uniform | darker at center | two-toned | mottled | streaked
- margin_lined_at_maturity: true if margin becomes striate with age
- central_depression: true if cap becomes funnel-shaped or depressed at center

STEM
- reticulation: none | partial (upper stem only) | full — key for boletes
- basal_mycelium_color: color of mycelium at base — e.g. white, lilac (diagnostic Laccaria)
- finger_stain_color: color left on fingers when rubbed — e.g. yellow (Retiboletus ornatipes)

VEIL
- cortina_present: true if a cortina (cobweb partial veil) is present — key Cortinarius feature

FLESH
- quantity: insubstantial/thin | moderate | thick

CHEMICAL REACTIONS (extract if present)
- KOH_cap / KOH_flesh: yellow, orange, red, negative, blackening
- FeSO4_cap / FeSO4_flesh: blue-green, pink, grey-green, negative

MICROSCOPIC (extract if present in source)
- basidia_spore_count: 4-spored | 2-spored | mixed

ECOLOGY
- trophic_mode: mycorrhizal | saprotrophic | parasitic
- growth_position: terrestrial | lignicolous (on wood) | coprophilous (on dung)

SAFETY
- edibility_status: MUST be one of: edible, choice, conditionally edible, inedible, toxic, deadly
- known_toxins: named toxins only e.g. ["amatoxins", "ibotenic acid", "muscimol", "gyromitrin"]"""

# ── Per-group contextual notes (subsets of _SYSTEM_PROMPT_NOTES) ──────────

_NOTES_PASS1 = """

FIELD-SPECIFIC NOTES

HYMENIUM
- Use "ridges" for species with forking/blunt ridges or "false gills" that run down
  the stem and fork repeatedly (e.g. Cantharellus, Craterellus). These are NOT true
  gills — they are blunt, vein-like, and cannot be separated from the cap flesh.
- Use "gills" only for true blade-like gills that can be cleanly separated.
- Use "smooth" for species with no distinct hymenophore structure (e.g. Craterellus
  cornucopioides interior, puffballs).

SAFETY
- edibility_status: MUST be one of: edible, choice, conditionally edible, inedible, toxic, deadly
- known_toxins: named toxins only e.g. ["amatoxins", "ibotenic acid", "muscimol", "gyromitrin"]"""

_NOTES_GROUP_A = """

FIELD-SPECIFIC NOTES

CAP
- colors: list of colors IN ENGLISH when fresh (translate from source language)
- color_faded: color when dry/faded IN ENGLISH
- color_pattern: uniform | darker at center | two-toned | mottled | streaked
- margin_lined_at_maturity: true if margin becomes striate with age
- central_depression: true if cap becomes funnel-shaped or depressed at center"""

_NOTES_GROUP_B = """

FIELD-SPECIFIC NOTES

GILLS — only populate gills fields when hymenium_type == "gills"
- color_with_age: e.g. "white to pink then brown", "yellow becoming rusty"
- thickness: thin | thick (thick gills are diagnostic e.g. for Laccaria)
- texture: waxy (Hygrocybe), brittle (Russula/Lactarius), normal/soft

PORES — only populate pores fields when hymenium_type == "pores"
- color: pore surface color when fresh
- bruising_color: "blue" (bluing), "slowly orangish-brown", "none"
- density_per_mm: e.g. "1-2 per mm", "3-4 per mm" """

_NOTES_GROUP_C = """

FIELD-SPECIFIC NOTES

STEM
- reticulation: none | partial (upper stem only) | full — key for boletes
- basal_mycelium_color: color of mycelium at base — e.g. white, lilac (diagnostic Laccaria)
- finger_stain_color: color left on fingers when rubbed — e.g. yellow (Retiboletus ornatipes)

VEIL
- cortina_present: true if a cortina (cobweb partial veil) is present — key Cortinarius feature"""

_NOTES_GROUP_D = """

FIELD-SPECIFIC NOTES

FLESH
- quantity: insubstantial/thin | moderate | thick

CHEMICAL REACTIONS (extract if present)
- KOH_cap / KOH_flesh: yellow, orange, red, negative, blackening
- FeSO4_cap / FeSO4_flesh: blue-green, pink, grey-green, negative"""

_NOTES_GROUP_E = """

FIELD-SPECIFIC NOTES

MICROSCOPIC (extract if present in source)
- basidia_spore_count: 4-spored | 2-spored | mixed

ECOLOGY
- trophic_mode: mycorrhizal | saprotrophic | parasitic
- growth_position: terrestrial | lignicolous (on wood) | coprophilous (on dung)"""


# ── Helper: build a complete system prompt for a group ────────────────────


def _build_group_prompt(vocab_keys: set[str], notes: str) -> str:
    """Assemble header + filtered canonical values + group-specific notes."""
    return (
        _SYSTEM_PROMPT_HEADER
        + "\n\nCANONICAL VALUES (use these exact terms when the text matches):"
        + _build_vocab_guidance(vocab_keys)
        + notes
    )


# ── Combine: header + auto-generated canonical values + contextual notes ────
# Monolithic prompt (kept for backward compat / tests)

SYSTEM_PROMPT = (
    _SYSTEM_PROMPT_HEADER
    + "\n\nCANONICAL VALUES (use these exact terms when the text matches):"
    + _build_vocab_guidance()
    + _SYSTEM_PROMPT_NOTES
)

# ── Per-group system prompts ──────────────────────────────────────────────

SYSTEM_PROMPT_PASS1 = _build_group_prompt(_PASS1_VOCAB_KEYS, _NOTES_PASS1)
SYSTEM_PROMPT_GROUP_A = _build_group_prompt(_GROUP_A_VOCAB_KEYS, _NOTES_GROUP_A)
SYSTEM_PROMPT_GROUP_B = _build_group_prompt(_GROUP_B_VOCAB_KEYS, _NOTES_GROUP_B)
SYSTEM_PROMPT_GROUP_C = _build_group_prompt(_GROUP_C_VOCAB_KEYS, _NOTES_GROUP_C)
SYSTEM_PROMPT_GROUP_D = _build_group_prompt(_GROUP_D_VOCAB_KEYS, _NOTES_GROUP_D)
SYSTEM_PROMPT_GROUP_E = _build_group_prompt(_GROUP_E_VOCAB_KEYS, _NOTES_GROUP_E)


def _pass1_context(pass1: Pass1IdentityFeatures) -> str:
    """Format Pass 1 results as context for Pass 2 user prompts."""
    parts = [f"Species: {pass1.scientific_name}"]
    if pass1.genus:
        parts.append(f"Genus: {pass1.genus}")
    if pass1.family:
        parts.append(f"Family: {pass1.family}")
    if pass1.overall_body_form:
        parts.append(f"Body form: {pass1.overall_body_form}")
    if pass1.hymenium_type:
        parts.append(f"Hymenium type: {pass1.hymenium_type}")
    if pass1.growth_habit:
        parts.append(f"Growth habit: {pass1.growth_habit}")
    return "\n".join(parts)


def merge_extraction_results(
    pass1: Pass1IdentityFeatures,
    cap: Pass2CapFeatures,
    hymenium: Pass2HymeniumFeatures,
    stem_veil: Pass2StemVeilFeatures,
    flesh_chem: Pass2FleshChemFeatures,
    spore_eco: Pass2SporeEcoFeatures,
) -> ExtractedSpeciesFeatures:
    """Merge 6 partial results into a single ExtractedSpeciesFeatures."""
    return ExtractedSpeciesFeatures(
        # Pass 1 — identity, taxonomy, body plan, safety
        scientific_name=pass1.scientific_name,
        species_epithet=pass1.species_epithet,
        common_names=pass1.common_names,
        synonyms=pass1.synonyms,
        kingdom=pass1.kingdom,
        phylum=pass1.phylum,
        order=pass1.order,
        family=pass1.family,
        genus=pass1.genus,
        overall_body_form=pass1.overall_body_form,
        overall_size_class=pass1.overall_size_class,
        growth_habit=pass1.growth_habit,
        edibility_status=pass1.edibility_status,
        known_toxins=pass1.known_toxins,
        known_lookalikes=pass1.known_lookalikes,
        extraction_notes=pass1.extraction_notes,
        # Pass 1 → hymenium sub-model (flat str → nested)
        hymenium=ExtractedHymeniumFeatures(type=pass1.hymenium_type),
        # Group A — cap
        cap=cap.cap,
        # Group B — hymenium details
        gills=hymenium.gills,
        pores=hymenium.pores,
        tubes=hymenium.tubes,
        spore_print_color=hymenium.spore_print_color,
        # Group C — stem, veil, volva
        stem=stem_veil.stem,
        veil=stem_veil.veil,
        volva=stem_veil.volva,
        # Group D — flesh, chemical
        flesh=flesh_chem.flesh,
        chemical=flesh_chem.chemical,
        # Group E — spore, microscopic, ecology
        spore=spore_eco.spore,
        microscopic=spore_eco.microscopic,
        ecology=spore_eco.ecology,
    )


def _run_pass2_group(
    response_model: type,
    system: str,
    group_label: str,
    scientific_name: str,
    source_name: str,
    text: str,
    context: str,
    extra_hint: str = "",
):
    """Run a single Pass 2 structured_completion call."""
    hint = f"\n{extra_hint}" if extra_hint else ""
    prompt = (
        f"Extract {group_label} features for '{scientific_name}' "
        f"from the following text.\n"
        f"Source: {source_name}\n\n"
        f"CONTEXT FROM PASS 1:\n{context}\n{hint}\n"
        f"SOURCE TEXT:\n{text}"
    )
    return structured_completion(
        prompt=prompt,
        response_model=response_model,
        system=system,
        temperature=0.1,
        max_tokens=2048,
    )


def extract_features_from_text(
    scientific_name: str,
    text: str,
    source_name: str = "unknown",
) -> ExtractedSpeciesFeatures:
    """Extract structured features from source text using 2-pass grouped LLM calls.

    Pass 1: Identity, taxonomy, body plan, safety (~18 fields).
    Pass 2: 5 parallel groups (cap, hymenium, stem/veil, flesh/chem, spore/eco).
    Results are merged into a single ExtractedSpeciesFeatures.
    """
    truncated = text[:_MAX_TEXT_CHARS]

    # ── Pass 1: Identity & Body Plan ──────────────────────────────────────
    pass1_prompt = (
        f"Extract identity, taxonomy, body plan, and safety features "
        f"for '{scientific_name}' from the following text.\n"
        f"Source: {source_name}\n\n"
        f"{truncated}"
    )
    pass1 = structured_completion(
        prompt=pass1_prompt,
        response_model=Pass1IdentityFeatures,
        system=SYSTEM_PROMPT_PASS1,
        temperature=0.1,
        max_tokens=1024,
    )
    logger.info(
        "Pass 1 complete for %s: body_form=%s, hymenium=%s",
        pass1.scientific_name,
        pass1.overall_body_form,
        pass1.hymenium_type,
    )

    # ── Pass 2: 5 parallel groups ─────────────────────────────────────────
    context = _pass1_context(pass1)

    # Conditional hint for Group B based on hymenium type
    ht = (pass1.hymenium_type or "").lower()
    if ht == "pores":
        group_b_hint = "Focus on pore and tube fields (gills are not applicable)."
    elif ht == "gills":
        group_b_hint = "Focus on gill fields (pores and tubes are not applicable)."
    else:
        group_b_hint = ""

    group_args = [
        (Pass2CapFeatures, SYSTEM_PROMPT_GROUP_A, "cap & surface", ""),
        (Pass2HymeniumFeatures, SYSTEM_PROMPT_GROUP_B, "hymenium detail", group_b_hint),
        (Pass2StemVeilFeatures, SYSTEM_PROMPT_GROUP_C, "stem, veil & volva", ""),
        (Pass2FleshChemFeatures, SYSTEM_PROMPT_GROUP_D, "flesh & chemistry", ""),
        (Pass2SporeEcoFeatures, SYSTEM_PROMPT_GROUP_E, "spore, microscopic & ecology", ""),
    ]

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [
            pool.submit(
                _run_pass2_group,
                response_model=model,
                system=system,
                group_label=label,
                scientific_name=scientific_name,
                source_name=source_name,
                text=truncated,
                context=context,
                extra_hint=hint,
            )
            for model, system, label, hint in group_args
        ]
        cap, hymenium, stem_veil, flesh_chem, spore_eco = [f.result() for f in futures]

    logger.info("Pass 2 complete for %s — merging results", scientific_name)
    return merge_extraction_results(pass1, cap, hymenium, stem_veil, flesh_chem, spore_eco)


def save_extraction(
    features: ExtractedSpeciesFeatures,
    source_url: str,
    source_text: str,
    source_name: str,
) -> SourceObservation:
    """
    Upsert extraction result into source_observations (Layer 1).

    If a row for (scientific_name, source_name) already exists,
    its features_json and metadata are updated (upsert policy).
    """
    text_hash = hashlib.sha256(source_text.encode()).hexdigest()
    features_dict = features.model_dump()
    model_string = f"{settings.llm_provider}/{settings.llm_model}"

    session = get_session()
    try:
        existing = (
            session.query(SourceObservation)
            .filter_by(scientific_name=features.scientific_name, source_name=source_name)
            .first()
        )
        if existing:
            existing.features_json = features_dict
            existing.source_url = source_url
            existing.source_text_hash = text_hash
            existing.extraction_model = model_string
            existing.extraction_notes = features.extraction_notes
            obs = existing
        else:
            obs = SourceObservation(
                scientific_name=features.scientific_name,
                source_name=source_name,
                source_url=source_url,
                source_text_hash=text_hash,
                features_json=features_dict,
                extraction_model=model_string,
                extraction_notes=features.extraction_notes,
            )
            session.add(obs)
        session.commit()
        session.refresh(obs)
        return obs
    finally:
        session.close()

"""Pre-merge normalization for reconciliation.

Deterministic, zero-LLM-cost transformations applied to features_json
before reconciliation:
  1. Alias → canonical for qualitative scalar fields (vocabulary-aware)
  2. Numeric range pre-merge: min-of-mins / max-of-maxes
  3. List union pre-merge with deduplication (month-ordered for fruiting_months)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_VOCAB_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "reference" / "morphological_vocabulary.yaml"
)

# Mapping: features_json dot-path → vocabulary YAML key
_FIELD_TO_VOCAB: dict[str, str] = {
    # Identity / overall
    "hymenium.type": "hymenium_type",
    "overall_body_form": "overall_body_form",
    "overall_size_class": "overall_size_class",
    "growth_habit": "growth_habit",
    "spore_print_color": "spore_print_color",
    # Cap
    "cap.shape": "cap_shape",
    "cap.surface_moisture": "cap_surface_moisture",
    "cap.scales_or_warts": "scales_or_warts",
    "cap.surface_texture": "surface_texture",
    "cap.margin_type": "cap_margin",
    "cap.color_pattern": "cap_color_pattern",
    # Gills
    "gills.attachment": "gill_attachment",
    "gills.spacing": "gill_spacing",
    "gills.edge_texture": "gill_edge_texture",
    "gills.thickness": "gill_thickness",
    "gills.texture": "gill_texture",
    # Stem
    "stem.surface_texture": "surface_texture",
    "stem.shape": "stem_shape",
    "stem.attachment_position": "stipe_attachment_position",
    "stem.hollow_or_solid": "stem_interior",
    "stem.reticulation": "stem_reticulation",
    "stem.consistency": "stem_consistency",
    # Veil / Volva
    "veil.type": "veil_type",
    "veil.shape": "ring_shape",
    "veil.ring_position": "ring_position",
    "veil.ring_mobility": "ring_mobility",
    "veil.ring_persistence": "ring_persistence",
    "volva.type": "volva_type",
    # Flesh
    "flesh.odor": "flesh_odor",
    "flesh.taste": "flesh_taste",
    "flesh.texture": "flesh_texture",
    "flesh.hyphal_structure": "flesh_hyphal_structure",
    "flesh.cap_stem_consistency": "flesh_cap_stem_consistency",
    "flesh.quantity": "flesh_quantity",
    "flesh.latex_color": "flesh_latex_color",
    # Chemical
    "chemical.KOH_cap": "chemical_KOH",
    "chemical.KOH_flesh": "chemical_KOH",
    "chemical.FeSO4_cap": "chemical_FeSO4",
    "chemical.FeSO4_flesh": "chemical_FeSO4",
    "chemical.NH4OH_cap": "chemical_NH4OH",
    "chemical.NH4OH_flesh": "chemical_NH4OH",
    # Spore
    "spore.shape": "spore_shape",
    "spore.ornamentation": "spore_ornamentation",
    "spore.amyloidity": "spore_amyloidity",
    "spore.color_in_KOH": "spore_color_in_KOH",
    # Microscopic
    "microscopic.basidia_spore_count": "basidia_spore_count",
    "microscopic.cheilocystidia_shape": "cheilocystidia_shape",
    "microscopic.pleurocystidia_shape": "pleurocystidia_shape",
    "microscopic.pileipellis_type": "pileipellis_type",
    "microscopic.pileipellis_terminal_cell_shape": "pileipellis_terminal_cell_shape",
    "microscopic.cystidia_color_in_KOH": "cystidia_color_in_KOH",
    # Ecology
    "ecology.trophic_mode": "trophic_mode",
    "ecology.substrate": "substrate",
    "ecology.altitude_notes": "altitude_zone",
    "ecology.growth_position": "growth_position",
}

# Module-level vocabulary cache (full YAML dict)
_vocab_cache: dict | None = None

# Module-level alias map cache: {vocab_key: {alias_lower: canonical}}
_alias_map: dict[str, dict[str, str]] | None = None

# Vocab keys excluded from enforcement (open-ended color/variant vocabularies)
_NO_ENFORCE: frozenset[str] = frozenset(
    {
        "spore_print_color",
        "chemical_FeSO4",
        "chemical_NH4OH",
        "cheilocystidia_shape",
        "pileipellis_type",
        "cystidia_color_in_KOH",
    }
)

_SEMANTIC_MATCH_THRESHOLD = 0.82

# Set to False in unit tests to skip model loading
_SEMANTIC_MATCHING: bool = True

# Lazy caches for canonical sets and embeddings
_canonical_sets: dict[str, frozenset] | None = None
_encoder: Any | None = None
_canonical_embeddings: dict[str, tuple[list[str], Any]] | None = None


def load_vocabulary() -> dict:
    """Load and cache the morphological vocabulary YAML. Public API for reuse."""
    global _vocab_cache
    if _vocab_cache is not None:
        return _vocab_cache
    with open(_VOCAB_PATH) as f:
        _vocab_cache = yaml.safe_load(f)
    return _vocab_cache


def _get_alias_map() -> dict[str, dict[str, str]]:
    """Return {vocab_key: {alias_lower: canonical}} — loaded once, then cached."""
    global _alias_map
    if _alias_map is not None:
        return _alias_map
    vocab = load_vocabulary()
    _alias_map = {}
    for key, entry in vocab.items():
        aliases = entry.get("aliases", {})
        if aliases:
            _alias_map[key] = {str(alias).lower(): str(canon) for alias, canon in aliases.items()}
    return _alias_map


def _get_canonical_sets() -> dict[str, frozenset]:
    global _canonical_sets
    if _canonical_sets is not None:
        return _canonical_sets
    vocab = load_vocabulary()
    _canonical_sets = {
        key: frozenset(str(c).lower() for c in entry.get("canonical_values", {}).keys())
        for key, entry in vocab.items()
        if entry.get("canonical_values")
    }
    return _canonical_sets


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        from config import settings

        _encoder = SentenceTransformer(settings.embedding_model)
    return _encoder


def _get_canonical_embeddings():
    global _canonical_embeddings
    if _canonical_embeddings is not None:
        return _canonical_embeddings
    from sentence_transformers.util import cos_sim as _  # noqa: F401 — ensure import works

    enc = _get_encoder()
    _canonical_embeddings = {}
    for vocab_key, terms_set in _get_canonical_sets().items():
        terms = sorted(terms_set)
        _canonical_embeddings[vocab_key] = (terms, enc.encode(terms, convert_to_tensor=True))
    return _canonical_embeddings


def _semantic_match(term: str, vocab_key: str) -> str | None:
    """Return the closest canonical term if cosine similarity ≥ threshold, else None."""
    from sentence_transformers.util import cos_sim

    embs = _get_canonical_embeddings()
    if vocab_key not in embs:
        return None
    terms, canon_embs = embs[vocab_key]
    enc = _get_encoder()
    term_emb = enc.encode(term, convert_to_tensor=True)
    sims = cos_sim(term_emb, canon_embs)[0]
    best_idx = int(sims.argmax())
    if float(sims[best_idx]) >= _SEMANTIC_MATCH_THRESHOLD:
        return terms[best_idx]
    return None


def normalize_features(features: dict) -> dict:
    """Return a deep copy of features with vocabulary normalization applied.

    1. Alias → canonical substitution (case-insensitive).
    2. Enforcement: non-canonical values are resolved via semantic matching or set to None.
       Fields in _NO_ENFORCE (open-ended color/variant vocabularies) are skipped.
    - Never mutates the input dict.
    """
    result = copy.deepcopy(features)
    alias_map = _get_alias_map()

    for dot_path, vocab_key in _FIELD_TO_VOCAB.items():
        if vocab_key not in alias_map:
            continue
        key_aliases = alias_map[vocab_key]

        parts = dot_path.split(".", 1)
        if len(parts) == 1:
            field = parts[0]
            val = result.get(field)
            if isinstance(val, str) and val.lower() in key_aliases:
                result[field] = key_aliases[val.lower()]
        else:
            parent, field = parts
            parent_dict = result.get(parent)
            if not isinstance(parent_dict, dict):
                continue
            val = parent_dict.get(field)
            if isinstance(val, str) and val.lower() in key_aliases:
                parent_dict[field] = key_aliases[val.lower()]

    # Enforcement pass: set non-canonical values to None (with optional semantic fallback)
    canonical_sets = _get_canonical_sets()
    for dot_path, vocab_key in _FIELD_TO_VOCAB.items():
        if vocab_key in _NO_ENFORCE or vocab_key not in canonical_sets:
            continue
        allowed = canonical_sets[vocab_key]
        parts = dot_path.split(".", 1)
        node = result if len(parts) == 1 else result.get(parts[0])
        field = parts[-1]
        if not isinstance(node, dict):
            continue
        val = node.get(field)
        if not isinstance(val, str) or val.lower() in allowed:
            continue  # None, already canonical, or not a string
        matched = _semantic_match(val, vocab_key) if _SEMANTIC_MATCHING else None
        node[field] = matched

    return result


# ---------------------------------------------------------------------------
# Numeric range pre-merge
# ---------------------------------------------------------------------------

# (parent_or_None, field, "min"/"max")
_NUMERIC_RANGE_FIELDS = [
    ("cap", "diameter_min_cm", "min"),
    ("cap", "diameter_max_cm", "max"),
    ("stem", "height_min_cm", "min"),
    ("stem", "height_max_cm", "max"),
    ("stem", "diameter_min_cm", "min"),
    ("stem", "diameter_max_cm", "max"),
    ("spore", "length_min_um", "min"),
    ("spore", "length_max_um", "max"),
    ("spore", "width_min_um", "min"),
    ("spore", "width_max_um", "max"),
    ("spore", "spine_length_um", "min"),
    ("spore", "spine_base_width_um", "min"),
]


def pre_merge_numeric_ranges(observations: list[dict]) -> dict:
    """Deterministically merge numeric range fields across source observations.

    Returns {parent: {field: merged_value}} for fields with at least one non-null
    value. Fields where all sources are null are omitted.
    """
    result: dict = {}
    for parent, field, direction in _NUMERIC_RANGE_FIELDS:
        values = []
        for obs in observations:
            container = obs.get(parent) if parent else obs
            if isinstance(container, dict):
                v = container.get(field)
                if v is not None:
                    values.append(v)
        if not values:
            continue
        merged = min(values) if direction == "min" else max(values)
        if parent not in result:
            result[parent] = {}
        result[parent][field] = merged
    return result


# ---------------------------------------------------------------------------
# List field pre-merge
# ---------------------------------------------------------------------------

_LIST_FIELDS = [
    (None, "common_names"),
    (None, "synonyms"),
    (None, "known_lookalikes"),
    (None, "known_toxins"),
    ("cap", "colors"),
    ("ecology", "associated_trees"),
    ("ecology", "fruiting_months"),
    ("ecology", "geographic_regions"),
]

_MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_MONTH_RANK = {m: i for i, m in enumerate(_MONTH_ORDER)}


def pre_merge_list_fields(observations: list[dict]) -> dict:
    """Deterministically merge list fields across source observations.

    Returns union of all non-null lists, deduplicated (first-seen order).
    fruiting_months is additionally sorted by calendar order.
    """
    result: dict = {}
    for parent, field in _LIST_FIELDS:
        seen: dict[str, None] = {}  # ordered set via dict keys
        for obs in observations:
            container = obs.get(parent) if parent else obs
            if not isinstance(container, dict):
                continue
            lst = container.get(field)
            if not isinstance(lst, list):
                continue
            for item in lst:
                key = str(item)
                seen[key] = None
        if not seen:
            continue
        merged = list(seen.keys())
        if field == "fruiting_months":
            merged.sort(key=lambda m: _MONTH_RANK.get(m, 99))
        if parent is None:
            result[field] = merged
        else:
            if parent not in result:
                result[parent] = {}
            result[parent][field] = merged
    return result

"""Soft-Jaccard similarity scorer using vocabulary similarity matrices.

Compares discrete feature values between two species using pre-computed
similarity scores from morphological_vocabulary.yaml. Handles three field
types: vocabulary scalars (_FIELD_TO_VOCAB), color fields (list and scalar),
and boolean fields.
"""

from __future__ import annotations

from ingestion.normalize import _FIELD_TO_VOCAB, load_vocabulary
from ingestion.rubric import _get_nested

# Optimized per-field weights from scripts/optimize_jaccard.py (50 trials, 620 edges)
# Best Recall@5 = 15.7% (pure Jaccard ranking). Fields not listed default to 1.0.
DEFAULT_JACCARD_WEIGHTS: dict[str, float] = {
    "hymenium.type": 3.504,
    "overall_body_form": 2.582,
    "overall_size_class": 2.000,
    "growth_habit": 0.006,
    "spore_print_color": 3.837,
    "cap.shape": 1.205,
    "cap.surface_moisture": 0.970,
    "cap.surface_texture": 0.106,
    "cap.margin_type": 2.563,
    "gills.attachment": 3.874,
    "gills.spacing": 4.072,
    "gills.edge_texture": 4.983,
    "gills.thickness": 4.933,
    "gills.texture": 3.904,
    "stem.surface_texture": 3.123,
    "stem.shape": 0.603,
    "stem.attachment_position": 0.049,
    "stem.hollow_or_solid": 2.750,
    "stem.reticulation": 3.187,
    "stem.consistency": 2.682,
    "veil.type": 1.460,
    "veil.shape": 2.280,
    "veil.ring_position": 4.160,
    "veil.ring_mobility": 2.504,
    "veil.ring_persistence": 3.129,
    "volva.type": 1.070,
    "flesh.odor": 4.619,
    "flesh.taste": 1.909,
    "flesh.texture": 3.004,
    "flesh.hyphal_structure": 3.364,
    "flesh.cap_stem_consistency": 3.848,
    "flesh.quantity": 4.411,
    "cap.colors": 3.838,
    "gills.color": 2.577,
    "stem.color": 2.494,
    "flesh.color": 0.013,
    "pores.color": 2.572,
    "cap.bruising_color": 0.941,
    "flesh.bruising_color": 3.584,
    "stem.bruising_color": 0.030,
    "pores.bruising_color": 4.132,
    "veil.present": 3.498,
    "volva.present": 1.118,
    "flesh.latex_presence": 2.644,
    "cap.central_depression": 0.064,
    "cap.margin_lined_at_maturity": 3.518,
    "veil.cortina_present": 2.771,
}

# Color fields → vocabulary key (not covered by _FIELD_TO_VOCAB)
_COLOR_FIELDS: dict[str, str] = {
    "cap.colors": "color_palette",
    "gills.color": "color_palette",
    "stem.color": "color_palette",
    "flesh.color": "color_palette",
    "pores.color": "color_palette",
    "cap.bruising_color": "bruising_color",
    "flesh.bruising_color": "bruising_color",
    "stem.bruising_color": "bruising_color",
    "pores.bruising_color": "bruising_color",
}

# Boolean fields — exact match: 1.0 if same, 0.0 if different
_BOOLEAN_FIELDS: list[str] = [
    "veil.present",
    "volva.present",
    "flesh.latex_presence",
    "cap.central_depression",
    "cap.margin_lined_at_maturity",
    "veil.cortina_present",
]


def _lookup_matrix_score(matrix: dict[str, float], term_a: str, term_b: str) -> float:
    """Look up similarity between two terms in a similarity matrix.

    Returns 1.0 for identical terms, tries both key orderings, 0.0 if not found.
    """
    if term_a == term_b:
        return 1.0
    key1 = f"{term_a}-{term_b}"
    if key1 in matrix:
        return matrix[key1]
    key2 = f"{term_b}-{term_a}"
    if key2 in matrix:
        return matrix[key2]
    return 0.0


def _normalize_color(color: str, aliases: dict[str, str]) -> str:
    """Normalize a color string via lowercased alias lookup."""
    return aliases.get(color.lower(), color.lower())


def _color_list_similarity(
    colors_a: list[str],
    colors_b: list[str],
    matrix: dict[str, float],
    aliases: dict[str, str],
) -> float:
    """Best-match bidirectional color list similarity.

    For each color in A, find max similarity against all colors in B, average.
    Do the reverse. Return the mean of both directions.
    """
    norm_a = [_normalize_color(c, aliases) for c in colors_a]
    norm_b = [_normalize_color(c, aliases) for c in colors_b]

    if not norm_a or not norm_b:
        return 0.0

    a_to_b = sum(max(_lookup_matrix_score(matrix, a, b) for b in norm_b) for a in norm_a) / len(
        norm_a
    )
    b_to_a = sum(max(_lookup_matrix_score(matrix, b, a) for a in norm_a) for b in norm_b) / len(
        norm_b
    )

    return (a_to_b + b_to_a) / 2.0


def compute_soft_jaccard(
    features_a: dict,
    features_b: dict,
    vocab: dict | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute soft-Jaccard similarity between two species feature dicts.

    Args:
        features_a: features_json of species A
        features_b: features_json of species B
        vocab: loaded morphological_vocabulary.yaml dict (loaded automatically if None)
        weights: optional per-field weights (default: uniform 1.0)

    Returns:
        (overall_score, per_field_breakdown) where per_field_breakdown maps
        field path to similarity score for each computable field.
    """
    if vocab is None:
        vocab = load_vocabulary()
    if weights is None:
        weights = DEFAULT_JACCARD_WEIGHTS.copy() if DEFAULT_JACCARD_WEIGHTS else {}

    field_scores: dict[str, float] = {}

    # 1. Vocabulary scalar fields
    for dot_path, vocab_key in _FIELD_TO_VOCAB.items():
        entry = vocab.get(vocab_key)
        if entry is None:
            continue
        matrix = entry.get("similarity_matrix")
        if matrix is None:
            continue

        val_a = _get_nested(features_a, dot_path)
        val_b = _get_nested(features_b, dot_path)
        if val_a is None or val_b is None:
            continue
        if not isinstance(val_a, str) or not isinstance(val_b, str):
            continue

        raw_aliases = entry.get("aliases", {})
        aliases = {str(k).lower(): str(v) for k, v in raw_aliases.items()}
        a_norm = aliases.get(val_a.lower(), val_a.lower())
        b_norm = aliases.get(val_b.lower(), val_b.lower())

        field_scores[dot_path] = _lookup_matrix_score(matrix, a_norm, b_norm)

    # 2. Color fields
    for dot_path, vocab_key in _COLOR_FIELDS.items():
        entry = vocab.get(vocab_key)
        if entry is None:
            continue
        matrix = entry.get("similarity_matrix", {})
        raw_aliases = entry.get("aliases", {})
        aliases = {str(k).lower(): str(v) for k, v in raw_aliases.items()}

        val_a = _get_nested(features_a, dot_path)
        val_b = _get_nested(features_b, dot_path)
        if val_a is None or val_b is None:
            continue

        if isinstance(val_a, list) and isinstance(val_b, list):
            if not val_a or not val_b:
                continue
            field_scores[dot_path] = _color_list_similarity(val_a, val_b, matrix, aliases)
        elif isinstance(val_a, str) and isinstance(val_b, str):
            a_norm = _normalize_color(val_a, aliases)
            b_norm = _normalize_color(val_b, aliases)
            field_scores[dot_path] = _lookup_matrix_score(matrix, a_norm, b_norm)

    # 3. Boolean fields
    for dot_path in _BOOLEAN_FIELDS:
        val_a = _get_nested(features_a, dot_path)
        val_b = _get_nested(features_b, dot_path)
        if val_a is None or val_b is None:
            continue
        field_scores[dot_path] = 1.0 if val_a == val_b else 0.0

    # 4. Weighted soft-Jaccard aggregation
    if not field_scores:
        return 0.0, field_scores

    total_w = 0.0
    weighted_sum = 0.0
    for field, score in field_scores.items():
        w = weights.get(field, 1.0)
        weighted_sum += w * score
        total_w += w

    overall = weighted_sum / total_w if total_w > 0 else 0.0
    return overall, field_scores

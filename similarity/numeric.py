"""
Numeric similarity: direct comparison of measurement fields.

Complements embedding-based similarity with exact numeric overlap/proximity
for fields like cap diameter, stem height, spore dimensions, etc.
"""

from ingestion.rubric import (
    NUMERIC_RANGE_FIELDS,
    NUMERIC_SINGLE_FIELDS,
    _get_nested,
)

# Size bins keyed by min_field path.
# Each value is an ordered list of (label, upper_bound).
# Values above the last upper_bound are still classified as the last label (fallback).
_SIZE_BINS: dict[str, list[tuple[str, float]]] = {
    "cap.diameter_min_cm": [("small", 5.0), ("medium", 15.0), ("large", 25.0)],
    "stem.height_min_cm": [("small", 4.0), ("medium", 12.0), ("large", 25.0)],
    "stem.diameter_min_cm": [("small", 0.3), ("medium", 1.0), ("large", 2.5)],
    "spore.length_min_um": [("small", 6.0), ("medium", 14.0), ("large", 25.0)],
    "spore.width_min_um": [("small", 4.0), ("medium", 9.0), ("large", 15.0)],
}


def range_to_category(
    min_val: float | None,
    max_val: float | None,
    bins: list[tuple[str, float]],
) -> str | None:
    """
    Convert a (min, max) range to a size category using the provided bins.

    Midpoint = (min + max) / 2. If only one end is available, use it as the midpoint.
    Returns None if both are None.
    """
    if min_val is None and max_val is None:
        return None

    if min_val is not None and max_val is not None:
        midpoint = (min_val + max_val) / 2
    elif min_val is not None:
        midpoint = min_val
    else:
        midpoint = max_val

    for label, upper_bound in bins:
        if midpoint <= upper_bound:
            return label

    return bins[-1][0]


def category_similarity(
    cat_a: str | None,
    cat_b: str | None,
    bins: list[tuple[str, float]],
) -> float | None:
    """
    Compute similarity score between two size categories.

    Same category → 1.0, adjacent → 0.5, opposite → 0.0.
    Returns None if either category is None.
    """
    if cat_a is None or cat_b is None:
        return None

    labels = [label for label, _ in bins]
    idx_a = labels.index(cat_a)
    idx_b = labels.index(cat_b)
    distance = abs(idx_a - idx_b)

    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.5
    else:
        return 0.0


def single_proximity(a: float, b: float, tolerance: float) -> float | None:
    """
    Compute proximity score for two single numeric values.

    Returns 1.0 when a == b, linearly decreasing to 0.0 when |a - b| >= tolerance.
    Returns None if either value is None.
    """
    if a is None or b is None:
        return None

    distance = abs(a - b)
    if distance >= tolerance:
        return 0.0

    return 1.0 - (distance / tolerance)


def _parse_numeric(value) -> float | None:
    """Try to extract a single float from a value that may be a string like '1-2 per mm'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        import re

        # Extract first number from strings like "1-2 per mm", "5-20mm", "3–8 µm"
        numbers = re.findall(r"[\d.]+", value)
        if numbers:
            # For ranges encoded as strings, take the midpoint
            vals = [float(n) for n in numbers if n]
            if len(vals) >= 2:
                return (vals[0] + vals[1]) / 2
            if vals:
                return vals[0]
    return None


def compute_numeric_similarity(features_a: dict, features_b: dict) -> float:
    """
    Compute overall numeric similarity between two species feature dicts.

    Averages all computable category and single-proximity scores.
    Returns 0.0 if no numeric comparisons are possible (all fields None).
    """
    scores: list[float] = []

    for min_field, max_field in NUMERIC_RANGE_FIELDS:
        a_min = _parse_numeric(_get_nested(features_a, min_field))
        a_max = _parse_numeric(_get_nested(features_a, max_field))
        b_min = _parse_numeric(_get_nested(features_b, min_field))
        b_max = _parse_numeric(_get_nested(features_b, max_field))

        bins = _SIZE_BINS[min_field]
        cat_a = range_to_category(a_min, a_max, bins)
        cat_b = range_to_category(b_min, b_max, bins)

        score = category_similarity(cat_a, cat_b, bins)
        if score is not None:
            scores.append(score)

    for field, tolerance in NUMERIC_SINGLE_FIELDS:
        a_val = _parse_numeric(_get_nested(features_a, field))
        b_val = _parse_numeric(_get_nested(features_b, field))

        score = single_proximity(a_val, b_val, tolerance)
        if score is not None:
            scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)

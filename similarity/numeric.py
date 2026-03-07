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


def range_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> float | None:
    """
    Compute overlap ratio between two numeric ranges.

    Returns 1.0 for identical ranges, 0.0 for disjoint ranges, and a
    proportional value in between. Returns None if any input is None.
    """
    if any(v is None for v in (a_min, a_max, b_min, b_max)):
        return None

    overlap_start = max(a_min, b_min)
    overlap_end = min(a_max, b_max)
    overlap = max(0.0, overlap_end - overlap_start)

    union_start = min(a_min, b_min)
    union_end = max(a_max, b_max)
    union = union_end - union_start

    if union == 0:
        # Both ranges are single identical points
        return 1.0

    return overlap / union


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

    Averages all computable range-overlap and single-proximity scores.
    Returns 0.0 if no numeric comparisons are possible (all fields None).
    """
    scores: list[float] = []

    for min_field, max_field in NUMERIC_RANGE_FIELDS:
        a_min = _get_nested(features_a, min_field)
        a_max = _get_nested(features_a, max_field)
        b_min = _get_nested(features_b, min_field)
        b_max = _get_nested(features_b, max_field)

        # Parse in case values are strings
        a_min = _parse_numeric(a_min)
        a_max = _parse_numeric(a_max)
        b_min = _parse_numeric(b_min)
        b_max = _parse_numeric(b_max)

        score = range_overlap(a_min, a_max, b_min, b_max)
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

"""Fuzzy color matching for evaluation.

Reuses the canonical ``color_palette.similarity_matrix`` from
``data/reference/morphological_vocabulary.yaml`` — the same matrix that
``similarity/jaccard.py`` uses for cross-species soft-color matching. By
plugging it into evaluation we can count perceptually-adjacent predictions
(red↔orange 0.6, yellow↔orange 0.7, olive↔green 0.6, etc.) as "close enough"
hits without inventing a new threshold scheme.

Inputs are expected to be canonical lowercase color tokens (the 12 cap_color
canonical values: blue / green / purple / olive / pink / black / red / orange
/ grey / yellow / white / brown). Strings are lowercased defensively so that
upstream casing variation does not cause false misses.
"""

from functools import lru_cache

from ingestion.normalize import load_vocabulary

DEFAULT_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _color_matrix() -> dict[str, float]:
    """Return the canonical color similarity matrix (cached)."""
    vocab = load_vocabulary()
    return vocab.get("color_palette", {}).get("similarity_matrix", {})


def color_similarity(a: str, b: str) -> float:
    """Return similarity in [0.0, 1.0] between two canonical color tokens.

    1.0 for identical colors, the matrix entry for known pairs, 0.0 otherwise.
    Tries both key orderings (``"a-b"`` and ``"b-a"``).
    """
    a = a.lower()
    b = b.lower()
    if a == b:
        return 1.0
    matrix = _color_matrix()
    return matrix.get(f"{a}-{b}", matrix.get(f"{b}-{a}", 0.0))


def fuzzy_color_score(pred: str | None, gt_set: set[str]) -> float:
    """Best similarity between *pred* and any color in *gt_set*.

    Returns 0.0 for ``pred is None`` or empty ``gt_set``.
    """
    if pred is None or not gt_set:
        return 0.0
    return max(color_similarity(pred, g) for g in gt_set)


def fuzzy_color_match(
    pred: str | None,
    gt_set: set[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """True if *pred* is similar to any color in *gt_set* at or above *threshold*."""
    return fuzzy_color_score(pred, gt_set) >= threshold

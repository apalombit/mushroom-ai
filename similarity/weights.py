"""
Similarity weight management.

Default weights from config (.env):
    morphological: 0.60
    ecological:    0.25
    taxonomic:     0.15

Users can override via API query parameters. Weights are normalized to sum to 1.0.

TODO:
    - [ ] Implement get_weights() with default + user override
    - [ ] Implement normalize_weights()
"""

from config import settings


class SimilarityWeights:
    """Encapsulates the three feature group weights."""

    def __init__(
        self,
        morphological: float | None = None,
        ecological: float | None = None,
        taxonomic: float | None = None,
    ):
        self.morphological = morphological if morphological is not None else settings.weight_morphological
        self.ecological = ecological if ecological is not None else settings.weight_ecological
        self.taxonomic = taxonomic if taxonomic is not None else settings.weight_taxonomic
        self._normalize()

    def _normalize(self):
        total = self.morphological + self.ecological + self.taxonomic
        if total > 0:
            self.morphological /= total
            self.ecological /= total
            self.taxonomic /= total

    def as_dict(self) -> dict[str, float]:
        return {
            "morphological": round(self.morphological, 3),
            "ecological": round(self.ecological, 3),
            "taxonomic": round(self.taxonomic, 3),
        }

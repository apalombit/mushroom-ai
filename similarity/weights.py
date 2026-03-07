"""
Similarity weight management.

Seven embedding/comparison groups + body-form gating toggle:
    macro_visual:    0.69  (cap shape/color, body form, spore print)
    structural:      0.12  (gills/pores, stem, veil, volva)
    flesh_sensory:   0.07  (flesh, odor, taste)
    microscopic_lab: 0.02  (spores, cystidia, chemical)
    ecological:      0.01  (habitat, season, trees)
    taxonomic:       0.01  (family, genus)
    numeric:         0.10  (measurement overlap)
    body_form_filter: False (exclude incompatible body forms)

All float weights are normalized to sum to 1.0.
Users can override via API query parameters.
Legacy `morphological=` kwarg distributes across the 4 morphological sub-groups.
"""

from config import settings

# Relative proportions for distributing a legacy morphological weight
# across the 4 sub-groups (must sum to 1.0)
_MORPH_DISTRIBUTION = {
    "macro_visual": 0.566,
    "structural": 0.283,
    "flesh_sensory": 0.094,
    "microscopic_lab": 0.057,
}

WEIGHT_FIELDS = (
    "macro_visual",
    "structural",
    "flesh_sensory",
    "microscopic_lab",
    "ecological",
    "taxonomic",
    "numeric",
)


class SimilarityWeights:
    """Encapsulates the seven feature group weights + body-form filter toggle."""

    def __init__(
        self,
        macro_visual: float | None = None,
        structural: float | None = None,
        flesh_sensory: float | None = None,
        microscopic_lab: float | None = None,
        ecological: float | None = None,
        taxonomic: float | None = None,
        numeric: float | None = None,
        body_form_filter: bool | None = None,
        *,
        morphological: float | None = None,
    ):
        # Legacy compat: distribute morphological across 4 sub-groups
        if morphological is not None:
            sub_args = {
                "macro_visual": macro_visual,
                "structural": structural,
                "flesh_sensory": flesh_sensory,
                "microscopic_lab": microscopic_lab,
            }
            for sub, ratio in _MORPH_DISTRIBUTION.items():
                if sub_args[sub] is None:
                    setattr(self, sub, morphological * ratio)
                else:
                    setattr(self, sub, sub_args[sub])
        else:
            self.macro_visual = _default(macro_visual, settings.weight_macro_visual)
            self.structural = _default(structural, settings.weight_structural)
            self.flesh_sensory = _default(flesh_sensory, settings.weight_flesh_sensory)
            self.microscopic_lab = _default(microscopic_lab, settings.weight_microscopic_lab)

        self.ecological = _default(ecological, settings.weight_ecological)
        self.taxonomic = _default(taxonomic, settings.weight_taxonomic)
        self.numeric = _default(numeric, settings.weight_numeric)
        self.body_form_filter = _default(body_form_filter, settings.weight_body_form_filter)
        self._normalize()

    def _normalize(self):
        total = sum(getattr(self, f) for f in WEIGHT_FIELDS)
        if total > 0:
            for f in WEIGHT_FIELDS:
                setattr(self, f, getattr(self, f) / total)

    def as_dict(self) -> dict[str, float | bool]:
        d: dict[str, float | bool] = {f: round(getattr(self, f), 4) for f in WEIGHT_FIELDS}
        d["body_form_filter"] = self.body_form_filter
        return d


def _default(value, fallback):
    return value if value is not None else fallback

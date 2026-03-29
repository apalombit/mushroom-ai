"""
Similarity weight management.

Weights are derived dynamically from the active GROUPING_PROFILE.
Each embedding group gets a weight, plus one for numeric similarity.
All float weights are normalized to sum to 1.0.
"""

from config import settings
from ingestion.rubric import EMBEDDING_GROUPS

WEIGHT_FIELDS = tuple(EMBEDDING_GROUPS.keys()) + ("numeric",)


class SimilarityWeights:
    """Encapsulates per-group embedding weights + numeric weight + body-form filter."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        numeric: float | None = None,
        body_form_filter: bool | None = None,
        hymenium_filter: bool | None = None,
        size_class_filter: bool | None = None,
        group_filter: bool | None = None,
        alpha: float | None = None,
        morpho_pool_required: bool | None = None,
        morphotype_prefilter: bool | None = None,
        dangerous_filter: bool | None = None,
    ):
        numeric_w = numeric if numeric is not None else settings.weight_numeric
        n_groups = len(EMBEDDING_GROUPS)
        default_per_group = (1.0 - numeric_w) / n_groups if n_groups > 0 else 0.0

        input_weights = weights or {}
        for g in EMBEDDING_GROUPS:
            settings_val = getattr(settings, f"weight_{g}", None)
            default = settings_val if settings_val is not None else default_per_group
            setattr(self, g, input_weights.get(g, default))
        self.numeric = numeric_w
        self.body_form_filter = (
            body_form_filter if body_form_filter is not None else settings.weight_body_form_filter
        )
        self.hymenium_filter = (
            hymenium_filter if hymenium_filter is not None else settings.weight_hymenium_filter
        )
        self.size_class_filter = (
            size_class_filter
            if size_class_filter is not None
            else settings.weight_size_class_filter
        )
        self.group_filter = group_filter if group_filter is not None else True
        # Blend factor: 1.0 = pure embedding, 0.0 = pure Jaccard
        self.alpha = alpha if alpha is not None else 0.5
        self.morpho_pool_required = (
            morpho_pool_required
            if morpho_pool_required is not None
            else settings.weight_morpho_pool_required
        )
        self.morphotype_prefilter = (
            morphotype_prefilter
            if morphotype_prefilter is not None
            else settings.weight_morphotype_prefilter
        )
        self.dangerous_filter = (
            dangerous_filter
            if dangerous_filter is not None
            else settings.weight_dangerous_filter
        )
        self._normalize()

    def _normalize(self):
        total = sum(getattr(self, f) for f in WEIGHT_FIELDS)
        if total > 0:
            for f in WEIGHT_FIELDS:
                setattr(self, f, getattr(self, f) / total)

    def as_dict(self) -> dict[str, float | bool]:
        d: dict[str, float | bool] = {f: round(getattr(self, f), 4) for f in WEIGHT_FIELDS}
        d["body_form_filter"] = self.body_form_filter
        d["hymenium_filter"] = self.hymenium_filter
        d["size_class_filter"] = self.size_class_filter
        d["group_filter"] = self.group_filter
        d["alpha"] = self.alpha
        d["morpho_pool_required"] = self.morpho_pool_required
        d["morphotype_prefilter"] = self.morphotype_prefilter
        d["dangerous_filter"] = self.dangerous_filter
        return d

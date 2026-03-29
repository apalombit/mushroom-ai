"""Unit tests for search gating logic (hymenium, size_class, morpho pool, dangerous)."""

from ingestion.rubric import MORPHOLOGICAL_GROUP_NAMES, _get_nested
from similarity.search import _DANGEROUS_EDIBILITY, _SAFE_EDIBILITY


class TestHymeniumGating:
    """Test the hymenium-type gate logic in isolation."""

    def _should_exclude(self, query_features, cand_features, filter_on=True):
        """Reproduce the gating logic from search_lookalikes."""
        if not filter_on:
            return False
        query_hymenium = _get_nested(query_features, "hymenium.type")
        if not query_hymenium:
            return False
        cand_hymenium = _get_nested(cand_features, "hymenium.type")
        if cand_hymenium and cand_hymenium.lower() != query_hymenium.lower():
            return True
        return False

    def test_excludes_mismatch(self):
        query = {"hymenium": {"type": "gills"}}
        cand = {"hymenium": {"type": "pores"}}
        assert self._should_exclude(query, cand) is True

    def test_keeps_match(self):
        query = {"hymenium": {"type": "gills"}}
        cand = {"hymenium": {"type": "gills"}}
        assert self._should_exclude(query, cand) is False

    def test_case_insensitive(self):
        query = {"hymenium": {"type": "Gills"}}
        cand = {"hymenium": {"type": "gills"}}
        assert self._should_exclude(query, cand) is False

    def test_keeps_null_candidate(self):
        """Null hymenium on candidate = no info, should pass (not be excluded)."""
        query = {"hymenium": {"type": "gills"}}
        cand = {}
        assert self._should_exclude(query, cand) is False

    def test_keeps_null_query(self):
        """Null hymenium on query = can't filter, should pass all."""
        query = {}
        cand = {"hymenium": {"type": "pores"}}
        assert self._should_exclude(query, cand) is False

    def test_disabled_keeps_mismatch(self):
        query = {"hymenium": {"type": "gills"}}
        cand = {"hymenium": {"type": "pores"}}
        assert self._should_exclude(query, cand, filter_on=False) is False


class TestSizeClassGating:
    """Test the size-class gate logic in isolation."""

    def _should_exclude(self, query_features, cand_features, filter_on=True):
        if not filter_on:
            return False
        query_size = _get_nested(query_features, "overall_size_class")
        if not query_size:
            return False
        cand_size = _get_nested(cand_features, "overall_size_class")
        if cand_size and cand_size.lower() != query_size.lower():
            return True
        return False

    def test_excludes_mismatch(self):
        query = {"overall_size_class": "large"}
        cand = {"overall_size_class": "small"}
        assert self._should_exclude(query, cand) is True

    def test_keeps_match(self):
        query = {"overall_size_class": "medium"}
        cand = {"overall_size_class": "medium"}
        assert self._should_exclude(query, cand) is False

    def test_keeps_null_candidate(self):
        query = {"overall_size_class": "large"}
        cand = {}
        assert self._should_exclude(query, cand) is False

    def test_keeps_null_query(self):
        query = {}
        cand = {"overall_size_class": "small"}
        assert self._should_exclude(query, cand) is False

    def test_disabled_keeps_mismatch(self):
        query = {"overall_size_class": "large"}
        cand = {"overall_size_class": "small"}
        assert self._should_exclude(query, cand, filter_on=False) is False


class TestMorphoPoolPriority:
    """Test that MORPHOLOGICAL_GROUP_NAMES correctly partitions groups."""

    def test_non_morpho_groups_excluded(self):
        for g in ("ecological", "habitat", "trees", "growth", "taxonomic"):
            assert g not in MORPHOLOGICAL_GROUP_NAMES

    def test_morpho_groups_present(self):
        # At least some morphological groups should exist
        assert len(MORPHOLOGICAL_GROUP_NAMES) > 0

    def test_morpho_pool_logic(self):
        """Simulate pool construction: eco-only candidate should be excluded."""
        # Mock group_maps: candidate 1 in cap_shape, candidate 2 only in ecological
        morpho_group = "cap_shape"
        assert morpho_group in MORPHOLOGICAL_GROUP_NAMES

        group_maps = {
            morpho_group: {1: ("Species A", 0.9)},
            "ecological": {1: ("Species A", 0.8), 2: ("Species B", 0.85)},
        }

        # Morpho pool: only IDs from morphological groups
        morpho_ids = set()
        for g in MORPHOLOGICAL_GROUP_NAMES:
            if g in group_maps:
                morpho_ids.update(group_maps[g])

        assert 1 in morpho_ids  # in cap_shape
        assert 2 not in morpho_ids  # only in ecological

    def test_morpho_pool_disabled_keeps_all(self):
        """When disabled, union all groups."""
        group_maps = {
            "cap_shape": {1: ("Species A", 0.9)},
            "ecological": {2: ("Species B", 0.85)},
        }
        all_ids = set()
        for gmap in group_maps.values():
            all_ids.update(gmap)

        assert 1 in all_ids
        assert 2 in all_ids


class TestDangerousFilter:
    """Test the dangerous-lookalikes edibility gate logic in isolation."""

    def _classify(self, edibility: str | None) -> str | None:
        if not edibility:
            return None
        e = edibility.lower().strip()
        if e in _SAFE_EDIBILITY:
            return "safe"
        if e in _DANGEROUS_EDIBILITY:
            return "dangerous"
        return None

    def _should_exclude(self, query_edibility, cand_edibility, filter_on=True):
        """Reproduce the gating logic from search_lookalikes."""
        if not filter_on:
            return False
        q_group = self._classify(query_edibility)
        if not q_group:
            return False
        c_group = self._classify(cand_edibility)
        if not c_group:
            return False
        return c_group == q_group

    def test_keeps_opposite_edible_vs_toxic(self):
        assert self._should_exclude("edible", "toxic") is False

    def test_excludes_same_group_both_safe(self):
        assert self._should_exclude("edible", "choice") is True

    def test_keeps_opposite_toxic_vs_edible(self):
        assert self._should_exclude("toxic", "edible") is False

    def test_excludes_same_group_both_dangerous(self):
        assert self._should_exclude("deadly", "toxic") is True

    def test_keeps_opposite_conditional_vs_inedible(self):
        assert self._should_exclude("conditionally edible", "inedible") is False

    def test_keeps_all_when_query_null(self):
        assert self._should_exclude(None, "toxic") is False

    def test_keeps_null_candidate(self):
        assert self._should_exclude("edible", None) is False

    def test_disabled_keeps_same_group(self):
        assert self._should_exclude("edible", "choice", filter_on=False) is False

    def test_case_insensitive(self):
        assert self._should_exclude("Edible", "TOXIC") is False
        assert self._should_exclude("Edible", "Choice") is True

    def test_unknown_edibility_passes(self):
        assert self._should_exclude("edible", "unknown") is False
        assert self._should_exclude("unknown", "toxic") is False

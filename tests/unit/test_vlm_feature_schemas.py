"""Tests for vlm_feature_schemas: describe-then-classify models + guardrails."""

import pytest

from vision.labeling.vlm_feature_schemas import (
    FEATURE_REGISTRY,
    CapColorResult,
    HymeniumTypeResult,
)

# ---------------------------------------------------------------------------
# HymeniumTypeResult guardrails
# ---------------------------------------------------------------------------


class TestHymeniumTypeGuardrails:
    def test_not_visible_clears_all(self):
        r = HymeniumTypeResult(
            visible=False,
            visual_description="some text",
            reasoning="some reason",
            hymenium_type="gills",
            confidence="high",
        )
        assert r.visible is False
        assert r.hymenium_type is None
        assert r.visual_description is None
        assert r.reasoning is None
        assert r.confidence == "cannot_tell"

    def test_cannot_tell_nulls_classification(self):
        r = HymeniumTypeResult(
            visible=True,
            visual_description="thin blades",
            reasoning="looks like gills",
            hymenium_type="gills",
            confidence="cannot_tell",
        )
        assert r.hymenium_type is None
        assert r.confidence == "cannot_tell"

    def test_classification_forces_visible(self):
        r = HymeniumTypeResult(
            visible=False,
            hymenium_type="pores",
            confidence="high",
        )
        # not visible → clears classification
        assert r.hymenium_type is None
        assert r.confidence == "cannot_tell"

    def test_valid_high_confidence(self):
        r = HymeniumTypeResult(
            visible=True,
            visual_description="Blade-like structures radiating from center",
            reasoning="Classic gill pattern",
            hymenium_type="gills",
            confidence="high",
        )
        assert r.hymenium_type == "gills"
        assert r.confidence == "high"
        assert r.visible is True

    def test_valid_low_confidence(self):
        r = HymeniumTypeResult(
            visible=True,
            visual_description="Faint ridges, partially obscured",
            reasoning="Could be ridges or gills",
            hymenium_type="ridges",
            confidence="low",
        )
        assert r.hymenium_type == "ridges"
        assert r.confidence == "low"

    def test_all_hymenium_classes_accepted(self):
        for cls in ["gills", "ridges", "pores", "teeth", "smooth", "gleba", "alveolate"]:
            r = HymeniumTypeResult(
                visible=True,
                visual_description=f"Looks like {cls}",
                reasoning=f"Matches {cls} description",
                hymenium_type=cls,
                confidence="high",
            )
            assert r.hymenium_type == cls

    def test_invalid_class_rejected(self):
        with pytest.raises(Exception):
            HymeniumTypeResult(
                visible=True,
                visual_description="something",
                reasoning="reason",
                hymenium_type="lamellae",
                confidence="high",
            )


# ---------------------------------------------------------------------------
# CapColorResult guardrails
# ---------------------------------------------------------------------------


class TestCapColorGuardrails:
    def test_not_visible_clears_all(self):
        r = CapColorResult(
            visible=False,
            visual_description="red surface",
            reasoning="clearly red",
            cap_color="red",
            confidence="high",
        )
        assert r.visible is False
        assert r.cap_color is None
        assert r.visual_description is None
        assert r.reasoning is None
        assert r.confidence == "cannot_tell"

    def test_cannot_tell_nulls_classification(self):
        r = CapColorResult(
            visible=True,
            visual_description="overexposed",
            reasoning="can't determine",
            cap_color="white",
            confidence="cannot_tell",
        )
        assert r.cap_color is None

    def test_valid_color(self):
        r = CapColorResult(
            visible=True,
            visual_description="Deep chestnut surface",
            reasoning="Classic brown",
            cap_color="brown",
            confidence="high",
        )
        assert r.cap_color == "brown"

    def test_all_12_colors_accepted(self):
        colors = [
            "white",
            "yellow",
            "orange",
            "red",
            "pink",
            "brown",
            "grey",
            "olive",
            "green",
            "purple",
            "blue",
            "black",
        ]
        for color in colors:
            r = CapColorResult(
                visible=True,
                visual_description=f"{color} cap",
                reasoning=f"Matches {color}",
                cap_color=color,
                confidence="high",
            )
            assert r.cap_color == color

    def test_invalid_color_rejected(self):
        with pytest.raises(Exception):
            CapColorResult(
                visible=True,
                visual_description="magenta surface",
                reasoning="not canonical",
                cap_color="magenta",
                confidence="high",
            )


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------


class TestFeatureRegistry:
    def test_registry_has_both_features(self):
        assert "hymenium_type" in FEATURE_REGISTRY
        assert "cap_color" in FEATURE_REGISTRY

    def test_registry_schema_field_consistency(self):
        for name, info in FEATURE_REGISTRY.items():
            schema_cls = info["schema"]
            field_name = info["field"]
            assert field_name in schema_cls.model_fields, (
                f"{name}: field '{field_name}' not in {schema_cls.__name__}"
            )

    def test_all_schemas_have_describe_then_classify_order(self):
        """Verify field ordering: visible → description → reasoning → class → confidence."""
        for name, info in FEATURE_REGISTRY.items():
            fields = list(info["schema"].model_fields.keys())
            assert fields[0] == "visible", f"{name}: first must be 'visible'"
            assert fields[1] == "visual_description", (
                f"{name}: second must be 'visual_description'"
            )
            assert fields[2] == "reasoning", f"{name}: third must be 'reasoning'"
            assert fields[-1] == "confidence", f"{name}: last must be 'confidence'"

"""Tests for vlm_feature_schemas: describe-then-classify models + guardrails."""

import pytest

from vision.labeling.vlm_feature_schemas import (
    FEATURE_REGISTRY,
    CapColorResult,
    CapShapeResult,
    HymeniumTypeResult,
    RingPresenceResult,
    StemShapeResult,
    SubstrateResult,
    SurfaceTextureResult,
    VolvaPresenceResult,
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
# RingPresenceResult guardrails
# ---------------------------------------------------------------------------


class TestRingPresenceGuardrails:
    def test_not_visible_clears_all(self):
        r = RingPresenceResult(
            visible=False,
            visual_description="skirt-like ring",
            reasoning="clear annulus",
            ring_presence="present",
            confidence="high",
        )
        assert r.visible is False
        assert r.ring_presence is None
        assert r.visual_description is None
        assert r.reasoning is None
        assert r.confidence == "cannot_tell"

    def test_cannot_tell_nulls_classification(self):
        r = RingPresenceResult(
            visible=True,
            visual_description="upper stem partly hidden",
            reasoning="might be a ring zone",
            ring_presence="present",
            confidence="cannot_tell",
        )
        assert r.ring_presence is None

    def test_valid_present(self):
        r = RingPresenceResult(
            visible=True,
            visual_description="membranous skirt hangs from upper stipe",
            reasoning="classic pendant annulus",
            ring_presence="present",
            confidence="high",
        )
        assert r.ring_presence == "present"

    def test_valid_absent(self):
        r = RingPresenceResult(
            visible=True,
            visual_description="upper stem is bare and uniform",
            reasoning="no ring or zone visible",
            ring_presence="absent",
            confidence="high",
        )
        assert r.ring_presence == "absent"

    def test_classification_forces_visible(self):
        r = RingPresenceResult(
            visible=False,
            ring_presence="present",
            confidence="high",
        )
        # not visible → clears classification
        assert r.ring_presence is None

    def test_invalid_class_rejected(self):
        with pytest.raises(Exception):
            RingPresenceResult(
                visible=True,
                ring_presence="maybe",
                confidence="high",
            )


# ---------------------------------------------------------------------------
# VolvaPresenceResult guardrails
# ---------------------------------------------------------------------------


class TestVolvaPresenceGuardrails:
    def test_not_visible_clears_all(self):
        r = VolvaPresenceResult(
            visible=False,
            visual_description="sac at base",
            reasoning="saccate volva",
            volva_presence="present",
            confidence="high",
        )
        assert r.volva_presence is None
        assert r.visual_description is None
        assert r.confidence == "cannot_tell"

    def test_cannot_tell_nulls_classification(self):
        r = VolvaPresenceResult(
            visible=True,
            visual_description="base obscured by leaves",
            reasoning="cannot tell",
            volva_presence="present",
            confidence="cannot_tell",
        )
        assert r.volva_presence is None

    def test_valid_present(self):
        r = VolvaPresenceResult(
            visible=True,
            visual_description="sac-like cup with free margin at the base",
            reasoning="classic saccate volva",
            volva_presence="present",
            confidence="high",
        )
        assert r.volva_presence == "present"

    def test_valid_absent(self):
        r = VolvaPresenceResult(
            visible=True,
            visual_description="plain tapered base, no sac or scales",
            reasoning="no volva structures",
            volva_presence="absent",
            confidence="high",
        )
        assert r.volva_presence == "absent"

    def test_invalid_class_rejected(self):
        with pytest.raises(Exception):
            VolvaPresenceResult(
                visible=True,
                volva_presence="saccate",  # full type, not binary presence
                confidence="high",
            )


# ---------------------------------------------------------------------------
# SubstrateResult guardrails
# ---------------------------------------------------------------------------


class TestSubstrateGuardrails:
    def test_not_visible_clears_all(self):
        r = SubstrateResult(
            visible=False,
            visual_description="bare soil",
            reasoning="ground is plain humus",
            substrate="soil",
            confidence="high",
        )
        assert r.visible is False
        assert r.substrate is None
        assert r.visual_description is None
        assert r.reasoning is None
        assert r.confidence == "cannot_tell"

    def test_cannot_tell_nulls_classification(self):
        r = SubstrateResult(
            visible=True,
            visual_description="base hidden by mulch",
            reasoning="cannot decide between woody debris and soil",
            substrate="soil",
            confidence="cannot_tell",
        )
        assert r.substrate is None

    def test_classification_forces_visible(self):
        r = SubstrateResult(
            visible=False,
            substrate="dead wood",
            confidence="high",
        )
        assert r.substrate is None
        assert r.confidence == "cannot_tell"

    def test_valid_high_confidence(self):
        r = SubstrateResult(
            visible=True,
            visual_description="bark and decayed grain visible at the base",
            reasoning="clearly a fallen log",
            substrate="dead wood",
            confidence="high",
        )
        assert r.substrate == "dead wood"
        assert r.confidence == "high"

    def test_all_substrate_classes_accepted(self):
        for cls in [
            "soil",
            "dead wood",
            "living tree",
            "leaf litter",
            "woody debris",
            "dung",
        ]:
            r = SubstrateResult(
                visible=True,
                visual_description=f"looks like {cls}",
                reasoning=f"matches {cls}",
                substrate=cls,
                confidence="high",
            )
            assert r.substrate == cls

    def test_invalid_class_rejected(self):
        with pytest.raises(Exception):
            SubstrateResult(
                visible=True,
                visual_description="moss",
                reasoning="moss-covered ground",
                substrate="moss",
                confidence="high",
            )


# ---------------------------------------------------------------------------
# SurfaceTextureResult guardrails
# ---------------------------------------------------------------------------


class TestSurfaceTextureGuardrails:
    def test_not_visible_clears_all(self):
        r = SurfaceTextureResult(
            visible=False,
            visual_description="silky sheen",
            reasoning="appressed fibrils",
            surface_texture="silky",
            confidence="high",
        )
        assert r.visible is False
        assert r.surface_texture is None
        assert r.confidence == "cannot_tell"

    def test_cannot_tell_nulls_classification(self):
        r = SurfaceTextureResult(
            visible=True,
            visual_description="cap soaking wet",
            reasoning="texture hidden",
            surface_texture="smooth",
            confidence="cannot_tell",
        )
        assert r.surface_texture is None

    def test_classification_forces_visible(self):
        r = SurfaceTextureResult(
            visible=False,
            surface_texture="scaly",
            confidence="high",
        )
        assert r.surface_texture is None
        assert r.confidence == "cannot_tell"

    def test_valid_high_confidence(self):
        r = SurfaceTextureResult(
            visible=True,
            visual_description="distinct flat scales",
            reasoning="scaly",
            surface_texture="scaly",
            confidence="high",
        )
        assert r.surface_texture == "scaly"

    def test_all_classes_accepted(self):
        for cls in [
            "smooth", "fibrillose", "silky", "velvety", "tomentose", "floccose",
            "scaly", "squarrose", "warty", "reticulate", "pruinose", "areolate",
            "pitted", "wrinkled",
        ]:
            r = SurfaceTextureResult(
                visible=True,
                visual_description=f"looks {cls}",
                reasoning=f"matches {cls}",
                surface_texture=cls,
                confidence="high",
            )
            assert r.surface_texture == cls

    def test_invalid_class_rejected(self):
        with pytest.raises(Exception):
            SurfaceTextureResult(
                visible=True,
                surface_texture="bald",  # alias, not canonical
                confidence="high",
            )


# ---------------------------------------------------------------------------
# StemShapeResult guardrails
# ---------------------------------------------------------------------------


class TestStemShapeGuardrails:
    def test_not_visible_clears_all(self):
        r = StemShapeResult(
            visible=False,
            visual_description="bulb at base",
            reasoning="distinct swelling",
            stem_shape="bulbous",
            confidence="high",
        )
        assert r.visible is False
        assert r.stem_shape is None
        assert r.confidence == "cannot_tell"

    def test_cannot_tell_nulls_classification(self):
        r = StemShapeResult(
            visible=True,
            visual_description="lower stem hidden in soil",
            reasoning="cannot judge base",
            stem_shape="equal",
            confidence="cannot_tell",
        )
        assert r.stem_shape is None

    def test_classification_forces_visible(self):
        r = StemShapeResult(
            visible=False,
            stem_shape="clavate",
            confidence="high",
        )
        assert r.stem_shape is None
        assert r.confidence == "cannot_tell"

    def test_valid_high_confidence(self):
        r = StemShapeResult(
            visible=True,
            visual_description="abrupt onion-shaped swelling at base",
            reasoning="bulbous",
            stem_shape="bulbous",
            confidence="high",
        )
        assert r.stem_shape == "bulbous"

    def test_all_classes_accepted(self):
        for cls in [
            "equal", "clavate", "obclavate", "bulbous", "attenuated",
            "ventricose", "compressed", "rooting",
        ]:
            r = StemShapeResult(
                visible=True,
                visual_description=f"looks {cls}",
                reasoning=f"matches {cls}",
                stem_shape=cls,
                confidence="high",
            )
            assert r.stem_shape == cls

    def test_invalid_class_rejected(self):
        with pytest.raises(Exception):
            StemShapeResult(
                visible=True,
                stem_shape="cylindrical",  # alias, not canonical
                confidence="high",
            )


# ---------------------------------------------------------------------------
# CapShapeResult guardrails
# ---------------------------------------------------------------------------


class TestCapShapeGuardrails:
    def test_not_visible_clears_all(self):
        r = CapShapeResult(
            visible=False,
            visual_description="rounded dome",
            reasoning="convex",
            cap_shape="convex",
            confidence="high",
        )
        assert r.visible is False
        assert r.cap_shape is None
        assert r.visual_description is None
        assert r.reasoning is None
        assert r.confidence == "cannot_tell"

    def test_cannot_tell_nulls_classification(self):
        r = CapShapeResult(
            visible=True,
            visual_description="top-down view, profile not visible",
            reasoning="cannot judge profile",
            cap_shape="convex",
            confidence="cannot_tell",
        )
        assert r.cap_shape is None

    def test_classification_forces_visible(self):
        r = CapShapeResult(
            visible=False,
            cap_shape="flat",
            confidence="high",
        )
        assert r.cap_shape is None
        assert r.confidence == "cannot_tell"

    def test_valid_high_confidence(self):
        r = CapShapeResult(
            visible=True,
            visual_description="distinct funnel-shaped cap with rim above center",
            reasoning="infundibuliform",
            cap_shape="infundibuliform",
            confidence="high",
        )
        assert r.cap_shape == "infundibuliform"

    def test_all_classes_accepted(self):
        for cls in [
            "convex", "flat", "conical", "campanulate", "parabolic",
            "umbonate", "depressed", "infundibuliform", "irregular",
            "globose", "ovoid",
        ]:
            r = CapShapeResult(
                visible=True,
                visual_description=f"looks {cls}",
                reasoning=f"matches {cls}",
                cap_shape=cls,
                confidence="high",
            )
            assert r.cap_shape == cls

    def test_invalid_class_rejected(self):
        with pytest.raises(Exception):
            CapShapeResult(
                visible=True,
                cap_shape="hemispherical",  # alias, not canonical
                confidence="high",
            )


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------


class TestFeatureRegistry:
    def test_registry_has_all_features(self):
        assert "hymenium_type" in FEATURE_REGISTRY
        assert "cap_color" in FEATURE_REGISTRY
        assert "ring_presence" in FEATURE_REGISTRY
        assert "volva_presence" in FEATURE_REGISTRY
        assert "substrate" in FEATURE_REGISTRY
        assert "surface_texture" in FEATURE_REGISTRY
        assert "stem_shape" in FEATURE_REGISTRY
        assert "cap_shape" in FEATURE_REGISTRY

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

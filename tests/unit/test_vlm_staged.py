"""Tests for two-stage chain-of-inquiry VLM extraction (Path C)."""

import json
from unittest.mock import patch

import pytest

from vision.labeling.vlm_feature_extractor import (
    _CONFIDENCE_RANK,
    extract_feature_batch,
    extract_feature_staged,
)
from vision.labeling.vlm_feature_schemas import HymeniumTypeResult
from vision.labeling.vlm_staged_schemas import (
    HYMENIUM_FAMILY_MEMBERS,
    STAGED_REGISTRY,
    HymeniumFamilyResult,
    HymeniumFeaturelessResult,
    HymeniumLinearResult,
    HymeniumPunctateResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_image(tmp_path):
    """Minimal dummy JPEG (header only)."""
    img = tmp_path / "test_image.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    return img


def _stage1_result(family="linear_radial", confidence="high", visible=True):
    return HymeniumFamilyResult(
        visible=visible,
        visual_description="radial blade-like structures",
        reasoning="they radiate from the stem like spokes",
        family=family,
        confidence=confidence,
    )


def _stage2_linear(hymenium_type="gills", confidence="high"):
    return HymeniumLinearResult(
        visible=True,
        visual_description="thin sharp blades stopping at the stem",
        reasoning="parallel knife-like edges, no decurrent attachment",
        hymenium_type=hymenium_type,
        confidence=confidence,
    )


def _stage2_punctate(hymenium_type="pores", confidence="high"):
    return HymeniumPunctateResult(
        visible=True,
        visual_description="surface with many tiny holes",
        reasoning="2-D openings, no shadows under the structures",
        hymenium_type=hymenium_type,
        confidence=confidence,
    )


def _stage2_featureless(hymenium_type="smooth", confidence="high"):
    return HymeniumFeaturelessResult(
        visible=True,
        visual_description="flat featureless crust on bark",
        reasoning="no discrete structures, no torn skin",
        hymenium_type=hymenium_type,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Registry coverage / config sanity
# ---------------------------------------------------------------------------


class TestStagedRegistry:
    def test_hymenium_registered(self):
        assert "hymenium_type" in STAGED_REGISTRY

    def test_hymenium_family_keys_match_registry(self):
        cfg = STAGED_REGISTRY["hymenium_type"]
        assert set(cfg.families) == set(HYMENIUM_FAMILY_MEMBERS)

    def test_hymenium_final_schema_matches_single_shot(self):
        """The final merged schema must equal the single-shot HymeniumTypeResult so
        downstream audit + analysis code keeps working."""
        cfg = STAGED_REGISTRY["hymenium_type"]
        assert cfg.final_schema is HymeniumTypeResult

    def test_family_members_cover_all_classes(self):
        """Every leaf hymenium class is reachable through exactly one family."""
        all_members: set[str] = set()
        for members in HYMENIUM_FAMILY_MEMBERS.values():
            assert not (members & all_members), "class assigned to two families"
            all_members |= members
        # Sanity: must include the 7 hymenium leaf classes
        assert all_members == {
            "gills", "ridges", "pores", "teeth", "alveolate", "smooth", "gleba"
        }


# ---------------------------------------------------------------------------
# extract_feature_staged — dispatch + abstain
# ---------------------------------------------------------------------------


class TestExtractFeatureStaged:
    def test_unknown_feature_raises(self, tmp_image):
        with pytest.raises(KeyError, match="unknown_feature"):
            extract_feature_staged(tmp_image, "unknown_feature")

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_routes_linear_radial_to_linear_stage2(self, mock_vlm, tmp_image):
        """family=linear_radial → stage 2 must use HymeniumLinearResult schema."""
        mock_vlm.side_effect = [
            _stage1_result(family="linear_radial"),
            _stage2_linear(hymenium_type="gills"),
        ]
        final, s1 = extract_feature_staged(tmp_image, "hymenium_type")

        # Two VLM calls
        assert mock_vlm.call_count == 2
        # Stage 2 used the linear schema
        s2_kwargs = mock_vlm.call_args_list[1].kwargs
        assert s2_kwargs["response_model"] is HymeniumLinearResult
        # Final result is the single-shot schema with the leaf class
        assert isinstance(final, HymeniumTypeResult)
        assert final.hymenium_type == "gills"
        assert s1.family == "linear_radial"

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_routes_punctate_to_punctate_stage2(self, mock_vlm, tmp_image):
        mock_vlm.side_effect = [
            _stage1_result(family="punctate"),
            _stage2_punctate(hymenium_type="teeth"),
        ]
        final, _ = extract_feature_staged(tmp_image, "hymenium_type")
        s2_kwargs = mock_vlm.call_args_list[1].kwargs
        assert s2_kwargs["response_model"] is HymeniumPunctateResult
        assert final.hymenium_type == "teeth"

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_routes_featureless_to_featureless_stage2(self, mock_vlm, tmp_image):
        mock_vlm.side_effect = [
            _stage1_result(family="featureless_or_internal"),
            _stage2_featureless(hymenium_type="gleba"),
        ]
        final, _ = extract_feature_staged(tmp_image, "hymenium_type")
        s2_kwargs = mock_vlm.call_args_list[1].kwargs
        assert s2_kwargs["response_model"] is HymeniumFeaturelessResult
        assert final.hymenium_type == "gleba"

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_stage1_cannot_tell_short_circuits(self, mock_vlm, tmp_image):
        """Stage 1 cannot_tell → only ONE VLM call, final is null + cannot_tell."""
        mock_vlm.return_value = _stage1_result(
            family=None, confidence="cannot_tell", visible=True
        )
        final, s1 = extract_feature_staged(tmp_image, "hymenium_type")
        assert mock_vlm.call_count == 1
        assert final.hymenium_type is None
        assert final.confidence == "cannot_tell"
        assert s1.family is None

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_stage1_visible_false_short_circuits(self, mock_vlm, tmp_image):
        """visible=False → enforced cannot_tell at stage 1, no stage 2 call."""
        mock_vlm.return_value = HymeniumFamilyResult(
            visible=False, family=None, confidence="cannot_tell"
        )
        final, _ = extract_feature_staged(tmp_image, "hymenium_type")
        assert mock_vlm.call_count == 1
        assert final.hymenium_type is None
        assert final.confidence == "cannot_tell"

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_stage2_cannot_tell_returns_null(self, mock_vlm, tmp_image):
        """Stage 1 routes correctly but stage 2 abstains → null leaf, both calls made."""
        mock_vlm.side_effect = [
            _stage1_result(family="linear_radial", confidence="high"),
            HymeniumLinearResult(visible=True, hymenium_type=None, confidence="cannot_tell"),
        ]
        final, _ = extract_feature_staged(tmp_image, "hymenium_type")
        assert mock_vlm.call_count == 2
        assert final.hymenium_type is None
        assert final.confidence == "cannot_tell"

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_min_confidence_merge_high_low(self, mock_vlm, tmp_image):
        """stage1=high + stage2=low → final low (min of both)."""
        mock_vlm.side_effect = [
            _stage1_result(family="punctate", confidence="high"),
            _stage2_punctate(hymenium_type="pores", confidence="low"),
        ]
        final, _ = extract_feature_staged(tmp_image, "hymenium_type")
        assert final.confidence == "low"

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_min_confidence_merge_low_high(self, mock_vlm, tmp_image):
        """stage1=low + stage2=high → final low."""
        mock_vlm.side_effect = [
            _stage1_result(family="linear_radial", confidence="low"),
            _stage2_linear(hymenium_type="ridges", confidence="high"),
        ]
        final, _ = extract_feature_staged(tmp_image, "hymenium_type")
        assert final.confidence == "low"

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_min_confidence_merge_high_high(self, mock_vlm, tmp_image):
        mock_vlm.side_effect = [
            _stage1_result(family="punctate", confidence="high"),
            _stage2_punctate(hymenium_type="alveolate", confidence="high"),
        ]
        final, _ = extract_feature_staged(tmp_image, "hymenium_type")
        assert final.confidence == "high"

    def test_confidence_rank_constants(self):
        """cannot_tell < low < high — guard the merge ordering."""
        assert (
            _CONFIDENCE_RANK["cannot_tell"]
            < _CONFIDENCE_RANK["low"]
            < _CONFIDENCE_RANK["high"]
        )


# ---------------------------------------------------------------------------
# extract_feature_batch — staged mode
# ---------------------------------------------------------------------------


class TestExtractFeatureBatchStaged:
    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_staged_writes_stage1_in_sidecar(self, mock_vlm, tmp_path):
        """Sidecar JSON must contain the top-level single-shot fields PLUS _stage1."""
        mock_vlm.side_effect = [
            _stage1_result(family="linear_radial"),
            _stage2_linear(hymenium_type="gills"),
        ]
        img = tmp_path / "img1.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        out_dir = tmp_path / "out"

        results = extract_feature_batch(
            image_paths=[img],
            feature_name="hymenium_type",
            out_dir=out_dir,
            staged=True,
        )

        assert len(results) == 1
        sidecar = out_dir / "hymenium_type" / "img1.json"
        data = json.loads(sidecar.read_text())
        # Top-level fields match single-shot shape (audit + analysis still work)
        assert data["hymenium_type"] == "gills"
        assert data["confidence"] == "high"
        # Stage 1 stashed for debugging
        assert "_stage1" in data
        assert data["_stage1"]["family"] == "linear_radial"

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_staged_unknown_feature_raises(self, mock_vlm, tmp_path):
        img = tmp_path / "img1.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        with pytest.raises(KeyError, match="No staged config"):
            extract_feature_batch(
                image_paths=[img],
                feature_name="cap_color",  # registered single-shot but not staged
                out_dir=tmp_path / "out",
                staged=True,
            )

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_staged_short_circuit_writes_one_call_sidecar(self, mock_vlm, tmp_path):
        """Stage 1 cannot_tell → still produces a valid sidecar with null leaf."""
        mock_vlm.return_value = _stage1_result(
            family=None, confidence="cannot_tell", visible=True
        )
        img = tmp_path / "img1.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        out_dir = tmp_path / "out"

        extract_feature_batch(
            image_paths=[img],
            feature_name="hymenium_type",
            out_dir=out_dir,
            staged=True,
        )
        # Only stage 1 was called
        assert mock_vlm.call_count == 1
        data = json.loads((out_dir / "hymenium_type" / "img1.json").read_text())
        assert data["hymenium_type"] is None
        assert data["confidence"] == "cannot_tell"
        assert data["_stage1"]["family"] is None

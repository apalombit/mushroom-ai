"""Tests for vlm_feature_extractor: extraction logic with mocked VLM."""

import json
from unittest.mock import patch

import pytest

from vision.labeling.vlm_feature_extractor import (
    extract_feature,
    extract_feature_batch,
)
from vision.labeling.vlm_feature_schemas import (
    CapColorResult,
    HymeniumTypeResult,
)


@pytest.fixture
def tmp_image(tmp_path):
    """Create a minimal dummy image file."""
    img = tmp_path / "test_image.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG header
    return img


@pytest.fixture
def tmp_images(tmp_path):
    """Create multiple dummy image files."""
    imgs = []
    for i in range(3):
        img = tmp_path / f"img_{i:03d}.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        imgs.append(img)
    return imgs


def _mock_hymenium_result(**overrides):
    defaults = {
        "visible": True,
        "visual_description": "Blade-like structures radiating from stem",
        "reasoning": "Classic gill pattern typical of agarics",
        "hymenium_type": "gills",
        "confidence": "high",
    }
    defaults.update(overrides)
    return HymeniumTypeResult(**defaults)


def _mock_cap_color_result(**overrides):
    defaults = {
        "visible": True,
        "visual_description": "Deep reddish-brown surface with lighter edges",
        "reasoning": "Dominant color is brown, chestnut-like",
        "cap_color": "brown",
        "confidence": "high",
    }
    defaults.update(overrides)
    return CapColorResult(**defaults)


# ---------------------------------------------------------------------------
# extract_feature
# ---------------------------------------------------------------------------


class TestExtractFeature:
    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_hymenium_type(self, mock_vlm, tmp_image):
        mock_vlm.return_value = _mock_hymenium_result()
        result = extract_feature(tmp_image, "hymenium_type")
        assert isinstance(result, HymeniumTypeResult)
        assert result.hymenium_type == "gills"
        mock_vlm.assert_called_once()

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_cap_color(self, mock_vlm, tmp_image):
        mock_vlm.return_value = _mock_cap_color_result()
        result = extract_feature(tmp_image, "cap_color")
        assert isinstance(result, CapColorResult)
        assert result.cap_color == "brown"

    def test_unknown_feature_raises(self, tmp_image):
        with pytest.raises(KeyError, match="unknown_feature"):
            extract_feature(tmp_image, "unknown_feature")

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_model_override_forwarded(self, mock_vlm, tmp_image):
        mock_vlm.return_value = _mock_hymenium_result()
        extract_feature(tmp_image, "hymenium_type", model_name="custom-model")
        _, kwargs = mock_vlm.call_args
        assert kwargs["model"] == "custom-model"

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_use_few_shot_false_skips_references(self, mock_vlm, tmp_image):
        """When use_few_shot=False, the call must use the single-image path
        even though reference_images/hymenium_type/ exists on disk."""
        mock_vlm.return_value = _mock_hymenium_result()
        extract_feature(tmp_image, "hymenium_type", use_few_shot=False)
        _, kwargs = mock_vlm.call_args
        # Single-image path: prompt is the user prompt and image_paths has one entry.
        assert kwargs["prompt"] != ""
        assert len(kwargs["image_paths"]) == 1
        assert "content_blocks" not in kwargs or kwargs.get("content_blocks") is None


# ---------------------------------------------------------------------------
# extract_feature_batch
# ---------------------------------------------------------------------------


class TestExtractFeatureBatch:
    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_batch_writes_sidecars(self, mock_vlm, tmp_images, tmp_path):
        mock_vlm.return_value = _mock_hymenium_result()
        out_dir = tmp_path / "output"

        results = extract_feature_batch(
            image_paths=tmp_images,
            feature_name="hymenium_type",
            out_dir=out_dir,
        )

        assert len(results) == 3
        for r in results:
            assert r["error"] is None
            assert r["skipped"] is False
            sidecar = out_dir / "hymenium_type" / f"{r['image_id']}.json"
            assert sidecar.exists()
            data = json.loads(sidecar.read_text())
            assert data["hymenium_type"] == "gills"

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_batch_skip_existing(self, mock_vlm, tmp_images, tmp_path):
        mock_vlm.return_value = _mock_hymenium_result()
        out_dir = tmp_path / "output"
        feature_dir = out_dir / "hymenium_type"
        feature_dir.mkdir(parents=True)

        # Pre-create sidecar for first image
        existing = feature_dir / f"{tmp_images[0].stem}.json"
        existing.write_text(json.dumps({"hymenium_type": "pores"}))

        results = extract_feature_batch(
            image_paths=tmp_images,
            feature_name="hymenium_type",
            out_dir=out_dir,
            skip_existing=True,
        )

        skipped = [r for r in results if r["skipped"]]
        assert len(skipped) == 1
        assert skipped[0]["image_id"] == tmp_images[0].stem
        # VLM called only for 2 non-skipped images
        assert mock_vlm.call_count == 2

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_batch_error_isolation(self, mock_vlm, tmp_images, tmp_path):
        """One failing image should not abort the batch."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("VLM connection failed")
            return _mock_hymenium_result()

        mock_vlm.side_effect = side_effect
        out_dir = tmp_path / "output"

        results = extract_feature_batch(
            image_paths=tmp_images,
            feature_name="hymenium_type",
            out_dir=out_dir,
        )

        errors = [r for r in results if r["error"] is not None]
        successes = [r for r in results if r["result"] is not None]
        assert len(errors) == 1
        assert len(successes) == 2
        # Error sidecar written
        err_sidecar = out_dir / "hymenium_type" / f"{errors[0]['image_id']}.json"
        assert err_sidecar.exists()
        assert "error" in json.loads(err_sidecar.read_text())

    @patch("vision.labeling.vlm_feature_extractor.structured_vision_completion")
    def test_custom_image_ids(self, mock_vlm, tmp_images, tmp_path):
        mock_vlm.return_value = _mock_cap_color_result()
        out_dir = tmp_path / "output"
        ids = ["alpha", "beta", "gamma"]

        results = extract_feature_batch(
            image_paths=tmp_images,
            feature_name="cap_color",
            out_dir=out_dir,
            image_ids=ids,
        )

        for r, expected_id in zip(results, ids):
            assert r["image_id"] == expected_id
            sidecar = out_dir / "cap_color" / f"{expected_id}.json"
            assert sidecar.exists()

    def test_mismatched_ids_raises(self, tmp_images, tmp_path):
        with pytest.raises(ValueError, match="image_ids length"):
            extract_feature_batch(
                image_paths=tmp_images,
                feature_name="hymenium_type",
                out_dir=tmp_path,
                image_ids=["only_one"],
            )

    def test_unknown_feature_raises(self, tmp_images, tmp_path):
        with pytest.raises(KeyError, match="bad_feature"):
            extract_feature_batch(
                image_paths=tmp_images,
                feature_name="bad_feature",
                out_dir=tmp_path,
            )

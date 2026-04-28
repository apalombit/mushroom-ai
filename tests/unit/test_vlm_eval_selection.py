"""Tests for the species-stratified dual-feature selection algorithm."""

import pandas as pd
import pytest
from PIL import Image

from vision.scripts.vlm_eval_run import (
    _center_crop_square,
    _dual_feature_select as _class_aware_select,
    _prepare_vlm_images,
)


def _make_test_df(
    species_classes: list[tuple[str, str, str, int]],
) -> pd.DataFrame:
    """Build a fake test DataFrame.

    Args:
        species_classes: List of (species, hymenium_type, cap_color, n_images).
    """
    rows = []
    img_counter = 0
    for sp, hy, cc, n in species_classes:
        for _ in range(n):
            rows.append(
                {
                    "image_id": f"img_{img_counter:04d}",
                    "species": sp,
                    "hymenium_type": hy,
                    "cap_color": cc,
                    "processed_path": f"/fake/{img_counter}.jpg",
                }
            )
            img_counter += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def balanced_df():
    """Small balanced dataset: 6 species, 3 hymenium, 3 cap_color."""
    return _make_test_df(
        [
            ("sp_a", "gills", "brown", 10),
            ("sp_b", "gills", "white", 10),
            ("sp_c", "pores", "yellow", 10),
            ("sp_d", "pores", "red", 10),
            ("sp_e", "teeth", "brown", 10),
            ("sp_f", "teeth", "white", 10),
        ]
    )


@pytest.fixture()
def skewed_df():
    """Skewed dataset: gills dominates, teeth/smooth are rare."""
    return _make_test_df(
        [
            ("sp_a", "gills", "brown", 50),
            ("sp_b", "gills", "white", 40),
            ("sp_c", "gills", "yellow", 30),
            ("sp_d", "pores", "brown", 20),
            ("sp_e", "pores", "red", 15),
            ("sp_f", "teeth", "orange", 5),
            ("sp_g", "smooth", "grey", 3),
            ("sp_h", "gills", "purple", 8),
        ]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassAwareSelect:
    def test_rare_class_coverage(self, skewed_df):
        """Rare hymenium classes (teeth, smooth) must be selected."""
        selected = _class_aware_select(skewed_df, target_n=100, random_state=42)

        selected_df = skewed_df[skewed_df["species"].isin(selected)]
        hy_classes = set(selected_df["hymenium_type"].unique())

        # All 4 hymenium classes should be covered
        assert "teeth" in hy_classes, "Rare class 'teeth' not selected"
        assert "smooth" in hy_classes, "Rare class 'smooth' not selected"
        assert "gills" in hy_classes
        assert "pores" in hy_classes

    def test_rare_color_coverage(self, skewed_df):
        """Rare cap colors should get coverage via pass 2."""
        selected = _class_aware_select(skewed_df, target_n=100, random_state=42)
        selected_df = skewed_df[skewed_df["species"].isin(selected)]
        colors = set(selected_df["cap_color"].unique())

        # orange and grey are each represented by only one species
        assert "orange" in colors, "Rare color 'orange' not selected"
        assert "grey" in colors, "Rare color 'grey' not selected"

    def test_target_n_respected(self):
        """Image count should be within reasonable range of target."""
        # Use a larger dataset where pass 3 has room to fill
        df = _make_test_df(
            [
                ("sp_a", "gills", "brown", 10),
                ("sp_b", "gills", "white", 10),
                ("sp_c", "gills", "yellow", 10),
                ("sp_d", "gills", "red", 10),
                ("sp_e", "pores", "brown", 10),
                ("sp_f", "pores", "white", 10),
                ("sp_g", "teeth", "orange", 8),
                ("sp_h", "smooth", "grey", 5),
                ("sp_i", "gills", "cream", 10),
                ("sp_j", "gills", "tan", 10),
                ("sp_k", "pores", "yellow", 10),
                ("sp_l", "gills", "brown", 10),
                ("sp_m", "gills", "white", 10),
                ("sp_n", "pores", "red", 10),
                ("sp_o", "gills", "brown", 10),
            ]
        )
        target = 80
        selected = _class_aware_select(df, target_n=target, random_state=42)
        selected_df = df[df["species"].isin(selected)]
        n = len(selected_df)

        # Must be at least enough for mandatory passes but not hugely over target
        assert n >= int(target * 0.5), f"Too few images: {n}"
        assert n <= int(target * 1.5), f"Too many images: {n} > {int(target * 1.5)}"

    def test_species_integrity(self, skewed_df):
        """All images of a selected species must be included (no partial)."""
        selected = _class_aware_select(skewed_df, target_n=100, random_state=42)
        selected_df = skewed_df[skewed_df["species"].isin(selected)]

        for sp in selected:
            sp_total = len(skewed_df[skewed_df["species"] == sp])
            sp_selected = len(selected_df[selected_df["species"] == sp])
            assert sp_total == sp_selected, (
                f"Species {sp}: {sp_selected}/{sp_total} images selected (should be all)"
            )

    def test_deterministic_with_seed(self, skewed_df):
        """Same seed → identical selection."""
        sel1 = _class_aware_select(skewed_df, target_n=100, random_state=42)
        sel2 = _class_aware_select(skewed_df, target_n=100, random_state=42)
        assert sel1 == sel2

    def test_different_seed_may_differ(self, skewed_df):
        """Different seeds can produce different selections."""
        sel1 = _class_aware_select(skewed_df, target_n=100, random_state=42)
        sel2 = _class_aware_select(skewed_df, target_n=100, random_state=99)
        # Not guaranteed to differ, but with enough species they usually do.
        # Just check both are valid (non-empty).
        assert len(sel1) > 0
        assert len(sel2) > 0

    def test_all_species_selected_when_target_exceeds_total(self, balanced_df):
        """If target_n > total images, select everything."""
        selected = _class_aware_select(balanced_df, target_n=1000, random_state=42)
        assert selected == set(balanced_df["species"].unique())

    def test_empty_df(self):
        """Empty DataFrame should return empty set."""
        empty = pd.DataFrame(
            columns=["image_id", "species", "hymenium_type", "cap_color", "processed_path"]
        )
        selected = _class_aware_select(empty, target_n=100, random_state=42)
        assert len(selected) == 0


# ---------------------------------------------------------------------------
# Center-crop preprocessing
# ---------------------------------------------------------------------------


class TestCenterCropSquare:
    def test_landscape_to_square(self):
        """A 1024x768 image becomes 768x768 (centered)."""
        img = Image.new("RGB", (1024, 768), color=(0, 0, 0))
        cropped = _center_crop_square(img)
        assert cropped.size == (768, 768)

    def test_portrait_to_square(self):
        """A 600x900 image becomes 600x600 (centered)."""
        img = Image.new("RGB", (600, 900), color=(0, 0, 0))
        cropped = _center_crop_square(img)
        assert cropped.size == (600, 600)

    def test_already_square_unchanged(self):
        """A square image stays the same size."""
        img = Image.new("RGB", (500, 500), color=(0, 0, 0))
        cropped = _center_crop_square(img)
        assert cropped.size == (500, 500)

    def test_centering_preserves_middle_pixels(self):
        """Crop must take the central region (not top-left corner)."""
        # Build a 4x2 image with a unique color in the geometric center.
        img = Image.new("RGB", (4, 2), color=(0, 0, 0))
        # Center two pixels at x=1,2 y=0,1 should be the kept square
        for x in (1, 2):
            for y in (0, 1):
                img.putpixel((x, y), (255, 0, 0))
        cropped = _center_crop_square(img)
        assert cropped.size == (2, 2)
        # The cropped 2x2 should be entirely red
        for x in range(2):
            for y in range(2):
                assert cropped.getpixel((x, y)) == (255, 0, 0)


class TestPrepareVlmImages:
    def test_default_returns_processed_paths(self, tmp_path):
        df = pd.DataFrame(
            [
                {"image_id": "a", "processed_path": str(tmp_path / "a.jpg")},
                {"image_id": "b", "processed_path": str(tmp_path / "b.jpg")},
            ]
        )
        paths = _prepare_vlm_images(df, vlm_resolution=None)
        assert paths == [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]

    def test_center_crop_requires_resolution(self):
        df = pd.DataFrame([{"image_id": "a", "processed_path": "/x"}])
        with pytest.raises(ValueError, match="vlm_resolution"):
            _prepare_vlm_images(df, vlm_resolution=None, square_strategy="center_crop")

    def test_unknown_strategy_raises(self):
        df = pd.DataFrame([{"image_id": "a", "processed_path": "/x"}])
        with pytest.raises(ValueError, match="square_strategy"):
            _prepare_vlm_images(df, vlm_resolution=896, square_strategy="bogus")

    def test_center_crop_writes_square_outputs(self, tmp_path, monkeypatch):
        """Integration: produces 896x896 square JPEGs in cache dir."""
        from vision.scripts import vlm_eval_run as mod

        # Redirect cache + raw dirs to tmp_path
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        proj_root = tmp_path
        monkeypatch.setattr(mod, "PROJECT_ROOT", proj_root)
        monkeypatch.setattr(mod, "RAW_DIR", raw_dir)

        # Make a 1024x768 raw image
        Image.new("RGB", (1024, 768), color=(120, 180, 90)).save(raw_dir / "img1.jpg")
        df = pd.DataFrame([{"image_id": "img1", "processed_path": "/missing"}])

        paths = _prepare_vlm_images(df, vlm_resolution=896, square_strategy="center_crop")

        assert len(paths) == 1
        out_path = proj_root / "data" / "images" / "processed" / "vlm_896_cc" / "img1.jpg"
        assert paths[0] == str(out_path)
        with Image.open(out_path) as im:
            assert im.size == (896, 896)

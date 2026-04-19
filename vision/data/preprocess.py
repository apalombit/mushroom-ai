"""ImagePreprocessor: raw images -> processed (versioned pipeline).

Deterministic preprocessing: RGB conversion, filtering, border removal, resize.
No augmentation — that happens at training time.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


@dataclass
class PreprocessResult:
    image_id: str
    status: str  # "ok", "skipped", "rejected"
    reject_reason: str | None = None


class ImagePreprocessor:
    """Configurable image preprocessor driven by a YAML pipeline config."""

    def __init__(self, config_path: str | Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.version = self.config["version"]

        filt = self.config["filtering"]
        self.min_short_side = filt["min_short_side_px"]
        self.max_aspect_ratio = filt["max_aspect_ratio"]
        self.reject_corrupt = filt["reject_corrupt"]

        res = self.config["resize"]
        self.target_short_side = res["target_short_side"]
        self.interpolation = getattr(Image, res["interpolation"])

        fmt = self.config["format"]
        self.jpeg_quality = fmt["jpeg_quality"]

        border = self.config["border_removal"]
        self.border_enabled = border["enabled"]
        self.border_threshold_std = border["threshold_std"]
        self.border_max_crop = border["max_crop_fraction"]

    def process(self, raw_path: Path, output_dir: Path) -> PreprocessResult:
        """Process a single raw image. Idempotent — skips if output exists."""
        image_id = raw_path.stem
        dest = output_dir / f"{image_id}.jpg"

        if dest.exists():
            return PreprocessResult(image_id=image_id, status="skipped")

        # Open and validate
        try:
            img = Image.open(raw_path)
            img.load()  # Force decode to catch corrupt files
        except Exception:
            if self.reject_corrupt:
                return PreprocessResult(
                    image_id=image_id, status="rejected", reject_reason="corrupt"
                )
            raise

        # Convert to RGB
        img = img.convert("RGB")

        # Size filtering
        w, h = img.size
        short_side = min(w, h)
        long_side = max(w, h)

        if short_side < self.min_short_side:
            return PreprocessResult(
                image_id=image_id,
                status="rejected",
                reject_reason=f"too_small ({w}x{h})",
            )

        if long_side / short_side > self.max_aspect_ratio:
            return PreprocessResult(
                image_id=image_id,
                status="rejected",
                reject_reason=f"bad_aspect ({long_side / short_side:.1f})",
            )

        # Border removal
        if self.border_enabled:
            img = self._remove_borders(img)

        # Resize shortest side
        img = self._resize_shortest_side(img)

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        img.save(dest, "JPEG", quality=self.jpeg_quality)

        return PreprocessResult(image_id=image_id, status="ok")

    def process_batch(self, raw_dir: Path, output_dir: Path) -> list[PreprocessResult]:
        """Process all images in raw_dir."""
        results = []
        raw_images = sorted(raw_dir.glob("*.jpg"))
        for raw_path in raw_images:
            result = self.process(raw_path, output_dir)
            results.append(result)
        return results

    def _remove_borders(self, img: Image.Image) -> Image.Image:
        """Remove constant-color borders by detecting low-variance edge strips."""
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        max_crop_h = int(h * self.border_max_crop)
        max_crop_w = int(w * self.border_max_crop)

        top = self._find_border_extent(arr, axis="top", max_crop=max_crop_h)
        bottom = self._find_border_extent(arr, axis="bottom", max_crop=max_crop_h)
        left = self._find_border_extent(arr, axis="left", max_crop=max_crop_w)
        right = self._find_border_extent(arr, axis="right", max_crop=max_crop_w)

        # Only crop if we'd still have a reasonable image
        new_h = h - top - bottom
        new_w = w - left - right
        if new_h < self.min_short_side or new_w < self.min_short_side:
            return img

        if top or bottom or left or right:
            return img.crop((left, top, w - right, h - bottom))
        return img

    def _find_border_extent(
        self, arr: np.ndarray, axis: str, max_crop: int
    ) -> int:
        """Find how many rows/cols from the given edge have std < threshold."""
        h, w = arr.shape[:2]
        extent = 0
        for i in range(max_crop):
            if axis == "top":
                strip = arr[i, :, :]
            elif axis == "bottom":
                strip = arr[h - 1 - i, :, :]
            elif axis == "left":
                strip = arr[:, i, :]
            elif axis == "right":
                strip = arr[:, w - 1 - i, :]
            else:
                break

            if np.std(strip) < self.border_threshold_std:
                extent += 1
            else:
                break
        return extent

    def _resize_shortest_side(self, img: Image.Image) -> Image.Image:
        """Resize so shortest side = target, preserving aspect ratio."""
        w, h = img.size
        short_side = min(w, h)
        if short_side <= self.target_short_side:
            return img  # Don't upscale

        scale = self.target_short_side / short_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), self.interpolation)

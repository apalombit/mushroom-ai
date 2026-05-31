"""Generate an HTML viewer comparing raw vs processed images for visual QA.

Usage:
    python -m vision.scripts.image_qa_viewer
    python -m vision.scripts.image_qa_viewer --n 500 --out data/image_qa_viewer.html
"""

import argparse
import base64
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402
from PIL import Image  # noqa: E402


def _img_to_base64(path: Path, max_side: int = 400) -> str:
    """Read image, resize for display, return base64 data URI."""
    img = Image.open(path).convert("RGB")
    # Resize for display (keep aspect ratio)
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _img_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def main():
    parser = argparse.ArgumentParser(description="Generate image QA HTML viewer")
    parser.add_argument("--n", type=int, default=300, help="Number of images to sample")
    parser.add_argument("--out", type=str, default="data/image_qa_viewer.html")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = yaml.safe_load(open(PROJECT_ROOT / "vision/config/training.yaml"))
    raw_dir = PROJECT_ROOT / config["paths"]["raw_images"]
    processed_dir = PROJECT_ROOT / config["paths"]["processed_images"] / "v1"

    # Find images that exist in both raw and processed
    raw_ids = {p.stem for p in raw_dir.glob("*.jpg")}
    processed_ids = {p.stem for p in processed_dir.glob("*.jpg")}
    common_ids = sorted(raw_ids & processed_ids)

    print(f"Raw: {len(raw_ids)}, Processed: {len(processed_ids)}, Common: {len(common_ids)}")

    n = min(args.n, len(common_ids))
    random.seed(args.seed)
    sampled = random.sample(common_ids, n)

    print(f"Sampling {n} images for viewer...")

    # Build HTML
    cards = []
    for i, image_id in enumerate(sampled):
        raw_path = raw_dir / f"{image_id}.jpg"
        proc_path = processed_dir / f"{image_id}.jpg"

        raw_b64 = _img_to_base64(raw_path)
        proc_b64 = _img_to_base64(proc_path)
        raw_w, raw_h = _img_dimensions(raw_path)
        proc_w, proc_h = _img_dimensions(proc_path)

        cards.append(f"""
        <div class="card">
            <div class="card-header">#{i + 1} — {image_id}</div>
            <div class="images">
                <div class="img-box">
                    <div class="label">Raw ({raw_w}x{raw_h})</div>
                    <img src="{raw_b64}" />
                </div>
                <div class="img-box">
                    <div class="label">Processed ({proc_w}x{proc_h})</div>
                    <img src="{proc_b64}" />
                </div>
            </div>
        </div>
        """)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} done")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Image QA Viewer — Raw vs Processed ({n} images)</title>
<style>
    body {{ font-family: -apple-system, sans-serif; background: #1a1a1a; color: #eee;
           margin: 0; padding: 20px; }}
    h1 {{ text-align: center; margin-bottom: 10px; }}
    .meta {{ text-align: center; color: #888; margin-bottom: 30px; }}
    .card {{ background: #2a2a2a; border-radius: 8px; margin-bottom: 16px;
             overflow: hidden; }}
    .card-header {{ padding: 8px 16px; background: #333; font-size: 13px;
                    font-family: monospace; color: #aaa; }}
    .images {{ display: flex; gap: 4px; padding: 8px; justify-content: center;
               flex-wrap: wrap; }}
    .img-box {{ text-align: center; }}
    .img-box img {{ max-height: 350px; max-width: 450px; border-radius: 4px; }}
    .label {{ font-size: 12px; color: #888; margin-bottom: 4px; }}
</style>
</head>
<body>
<h1>Image QA: Raw vs Processed</h1>
<div class="meta">
    Preprocessing: v1 (resize shortest_side=256, border removal, JPEG q95)<br>
    DINOv2 input: Resize(256) &rarr; CenterCrop(224) &rarr; ImageNet normalize (at embed time)<br>
    {n} random images shown
</div>
{"".join(cards)}
</body>
</html>"""

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"\nViewer saved to: {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

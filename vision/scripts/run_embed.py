"""CLI: processed images -> cached DINOv2 embeddings (.pt vectors).

Usage:
    python -m vision.scripts.run_embed
    python -m vision.scripts.run_embed --batch-size 16
"""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from vision.models.embed import EmbeddingExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 embeddings")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    config = yaml.safe_load(open(PROJECT_ROOT / "vision/config/training.yaml"))
    backbone = config["backbone"]
    device = config["device"]

    # Resolve paths: processed/v1 → embeddings/{backbone_slug}/processed-v1
    processed_dir = PROJECT_ROOT / config["paths"]["processed_images"] / "v1"
    backbone_slug = backbone.replace("/", "-")
    output_dir = PROJECT_ROOT / config["paths"]["embeddings"] / backbone_slug / "processed-v1"

    print(f"Backbone: {backbone}")
    print(f"Device: {device}")
    print(f"Input: {processed_dir}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {args.batch_size}\n")

    if not processed_dir.exists():
        print(f"No processed images at {processed_dir}")
        return

    extractor = EmbeddingExtractor(model_name=backbone, device=device)

    stats = extractor.extract_batch(
        processed_dir=processed_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
    )

    print(f"\nDone: {stats['new']} new embeddings, {stats['skipped']} skipped")
    print(f"Total: {stats['total']} images")
    print(f"Embeddings at: {output_dir}")


if __name__ == "__main__":
    main()

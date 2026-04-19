"""CLI: raw images -> processed (versioned preprocessing pipeline).

Usage:
    python -m vision.scripts.run_preprocess
    python -m vision.scripts.run_preprocess --config vision/config/pipeline_configs/preprocess_v1.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from db.connection import get_session
from vision.data.preprocess import ImagePreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw images")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to preprocess config YAML (default: preprocess_v1.yaml)",
    )
    args = parser.parse_args()

    training_config = yaml.safe_load(
        open(PROJECT_ROOT / "vision/config/training.yaml")
    )

    # Resolve config path
    if args.config:
        config_path = Path(args.config)
    else:
        config_dir = PROJECT_ROOT / training_config["paths"]["pipeline_configs"]
        config_path = config_dir / "preprocess_v1.yaml"

    preprocessor = ImagePreprocessor(config_path)

    raw_dir = PROJECT_ROOT / training_config["paths"]["raw_images"]
    output_dir = (
        PROJECT_ROOT
        / training_config["paths"]["processed_images"]
        / preprocessor.version
    )

    print(f"Config: {config_path}")
    print(f"Version: {preprocessor.version}")
    print(f"Input: {raw_dir}")
    print(f"Output: {output_dir}\n")

    if not raw_dir.exists():
        print(f"No raw images directory at {raw_dir}")
        return

    results = preprocessor.process_batch(raw_dir, output_dir)

    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skipped")
    rejected = [r for r in results if r.status == "rejected"]

    # Log rejections to image_quality table
    if rejected:
        session = get_session()
        for r in rejected:
            session.execute(
                text(
                    "INSERT INTO image_quality (image_id, is_verified, exclude_reason) "
                    "VALUES (:id, FALSE, :reason) "
                    "ON CONFLICT (image_id) DO UPDATE SET exclude_reason = :reason"
                ),
                {"id": r.image_id, "reason": r.reject_reason},
            )
        session.commit()
        session.close()

    print(f"Done: {ok} processed, {skipped} skipped, {len(rejected)} rejected")
    for r in rejected:
        print(f"  rejected: {r.image_id} — {r.reject_reason}")
    print(f"Output at: {output_dir}")


if __name__ == "__main__":
    main()

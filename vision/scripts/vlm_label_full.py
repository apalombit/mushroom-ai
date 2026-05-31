"""CLI: grade all processed images with a VLM for dataset cleanup.

Runs the same VLM labeling as the pilot but at full scale (~14k images),
persisting results to the DB for downstream manifest filtering.  Supports
resume — already-graded images are skipped automatically.

Usage:
    # Default (gemma4:31b-cloud, all images)
    python -m vision.scripts.vlm_label_full --out data/vlm_full/run_001

    # Smoke test (10 images only)
    python -m vision.scripts.vlm_label_full --out data/vlm_full/run_001 --limit 10

    # Different model
    python -m vision.scripts.vlm_label_full --out data/vlm_full/run_001 --model qwen3.5:cloud
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text  # noqa: E402
from tqdm import tqdm  # noqa: E402

from db.connection import get_session  # noqa: E402
from vision.labeling.vlm_db import get_graded_image_ids, persist_vlm_annotation  # noqa: E402
from vision.labeling.vlm_labeler import label_image  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "images" / "processed" / "v1"


def _fetch_all_image_ids(session) -> list[str]:
    """Get all image_ids from image_registry."""
    rows = session.execute(text("SELECT image_id FROM image_registry")).fetchall()
    return [r[0] for r in rows]


def _filter_to_existing(image_ids: list[str]) -> list[tuple[str, Path]]:
    """Keep only ids whose processed file exists on disk."""
    result = []
    for iid in image_ids:
        path = PROCESSED_DIR / f"{iid}.jpg"
        if path.exists():
            result.append((iid, path))
    return result


def main():
    parser = argparse.ArgumentParser(description="VLM full-scale image grading")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for JSON sidecars (e.g. data/vlm_full/run_001)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma4:31b-cloud",
        help="VLM model name (default: gemma4:31b-cloud)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="DB commit interval (default: 50)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of images to grade (for testing)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fetch all image ids and filter to existing processed files
    print("Loading image registry...")
    with get_session() as session:
        all_ids = _fetch_all_image_ids(session)
    print(f"  total image_ids in registry: {len(all_ids)}")

    candidates = _filter_to_existing(all_ids)
    print(f"  with processed v1 file:      {len(candidates)}")

    # Resume support: skip already-graded images
    with get_session() as session:
        already_graded = get_graded_image_ids(session)
    candidates = [(iid, p) for iid, p in candidates if iid not in already_graded]
    print(f"  already graded (skipped):    {len(already_graded)}")
    print(f"  to grade this run:           {len(candidates)}")

    if args.limit:
        candidates = candidates[: args.limit]
        print(f"  limited to:                  {len(candidates)}")

    if not candidates:
        print("\nNothing to grade — all images already done.")
        return

    # Grade images
    print(f"\nRunning VLM ({args.model})...")
    n_ok = 0
    n_err = 0

    with get_session() as session:
        for i, (image_id, image_path) in enumerate(
            tqdm(candidates, desc="VLM grading", unit="img")
        ):
            sidecar = out_dir / f"{image_id}.json"

            try:
                annotation = label_image(image_path, model_name=args.model)
                persist_vlm_annotation(session, image_id, annotation, model_name=args.model)
                payload = annotation.model_dump()
                sidecar.write_text(json.dumps(payload, indent=2))
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                sidecar.write_text(json.dumps({"error": err}, indent=2))
                n_err += 1

            # Periodic commit
            if (i + 1) % args.batch_size == 0:
                session.commit()

        # Final commit
        session.commit()

    print(f"\nDone. {n_ok} ok, {n_err} errors, {len(already_graded)} skipped.")
    print(f"  sidecars: {out_dir}")


if __name__ == "__main__":
    main()

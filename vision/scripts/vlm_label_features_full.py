"""Run a per-feature VLM extractor over the full visibility-gated image pool.

Produces one JSON sidecar per image at
``data/vlm_labels/<feature_name>/<image_id>.json`` for downstream ingestion
via ``vision.scripts.ingest_vlm_labels``.

Differences from ``vlm_eval_run.py``:
- No species-stratified sampling — runs on the FULL candidate pool.
- No ground-truth join; this is teacher-label production, not evaluation.
- Re-uses the same visibility filtering as training manifests
  (``_build_filtered_query``), so we only spend cloud cycles on images where
  the feature is actually visible.

Usage:
    .venv/bin/python -m vision.scripts.vlm_label_features_full \\
        --feature hymenium_type \\
        --vlm-resolution 896 --square center_crop --no-few-shot

Resumable: existing sidecars are skipped automatically (idempotent).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from db.connection import get_session  # noqa: E402
from vision.data.manifest import _build_filtered_query, build_full_pool_query  # noqa: E402
from vision.labeling.vlm_feature_extractor import extract_feature_batch  # noqa: E402
from vision.labeling.vlm_feature_schemas import FEATURE_REGISTRY  # noqa: E402
from vision.labeling.vlm_staged_schemas import STAGED_REGISTRY  # noqa: E402
from vision.scripts.vlm_eval_run import _center_crop_square  # noqa: E402

PROCESSED_DIR_DEFAULT = PROJECT_ROOT / "data" / "images" / "processed" / "v1"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "vlm_labels"


def _candidate_image_ids(session, feature: str, full_pool: bool = False) -> pd.DataFrame:
    """Return image_id + species for all visibility-gated candidates.

    When ``full_pool`` is False (default), restricts to images of species that
    had this feature extracted into ``features_json`` (i.e. a species_propagated
    annotation row exists). When True, returns the full visibility-gated pool
    regardless of species_propagated coverage — useful for teacher-labelling
    where we want the VLM to confirm absent on species the rubric ignores.
    """
    if full_pool:
        query, params = build_full_pool_query(feature)
        rows = session.execute(query, params).fetchall()
        return pd.DataFrame(rows, columns=["image_id", "species"])
    query, params = _build_filtered_query(feature, annotation_type="species_propagated")
    rows = session.execute(query, params).fetchall()
    return pd.DataFrame(rows, columns=["image_id", "species", "_gt"]).drop(columns=["_gt"])


def _prepare_images(
    df: pd.DataFrame,
    src_dir: Path,
    cache_dir: Path,
    resolution: int,
    square: str,
) -> list[tuple[str, Path]]:
    """Resize to `resolution` and optionally square-pad/center-crop.

    Returns list of (image_id, processed_path) for files that exist on disk.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[str, Path]] = []
    for _, row in df.iterrows():
        iid = row["image_id"]
        src = src_dir / f"{iid}.jpg"
        if not src.exists():
            continue
        dst = cache_dir / f"{iid}.jpg"
        if not dst.exists():
            img = Image.open(src).convert("RGB")
            if square == "center_crop":
                img = _center_crop_square(img)
            img = img.resize((resolution, resolution), Image.LANCZOS)
            img.save(dst, "JPEG", quality=92)
        out.append((iid, dst))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Per-feature VLM labelling at full scale")
    p.add_argument(
        "--feature", required=True, help=f"One of: {sorted(FEATURE_REGISTRY)}"
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Root output dir (sidecars → <out-root>/<feature>/)",
    )
    p.add_argument(
        "--src-dir",
        type=Path,
        default=PROCESSED_DIR_DEFAULT,
        help="Source dir for v1 processed images",
    )
    p.add_argument(
        "--vlm-resolution",
        type=int,
        default=896,
        help="Target inference resolution (default: 896)",
    )
    p.add_argument(
        "--square",
        choices=["center_crop", "none"],
        default="center_crop",
    )
    p.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot reference images (Path A default)",
    )
    p.add_argument(
        "--staged",
        action="store_true",
        help="Use staged extraction (only if registered in STAGED_REGISTRY)",
    )
    p.add_argument("--model", default=None, help="VLM model override")
    p.add_argument("--limit", type=int, default=None, help="Cap candidates (smoke test)")
    p.add_argument(
        "--full-pool",
        action="store_true",
        help=(
            "Drop the species_propagated requirement; label every "
            "visibility-gated image (broadens absent class coverage for "
            "features only extracted on a subset of species, e.g. volva)."
        ),
    )
    args = p.parse_args()

    if args.staged and args.feature not in STAGED_REGISTRY:
        raise SystemExit(
            f"--staged requested but {args.feature!r} not in STAGED_REGISTRY: "
            f"{sorted(STAGED_REGISTRY)}"
        )

    pool_label = "full visibility-gated pool" if args.full_pool else "species_propagated pool"
    print(f"Loading {pool_label} for {args.feature}...")
    with get_session() as session:
        df = _candidate_image_ids(session, args.feature, full_pool=args.full_pool)
    print(f"  candidates: {len(df)} images / {df['species'].nunique()} species")

    if args.limit:
        df = df.head(args.limit)
        print(f"  limited to first {len(df)}")

    cache_dir = (
        PROJECT_ROOT
        / "data"
        / "images"
        / "processed"
        / f"vlm_{args.vlm_resolution}_{args.square if args.square != 'none' else 'plain'}"
    )
    print(f"Preparing {args.vlm_resolution}px images in {cache_dir}...")
    prepared = _prepare_images(df, args.src_dir, cache_dir, args.vlm_resolution, args.square)
    print(f"  prepared: {len(prepared)} images on disk")

    image_ids = [iid for iid, _ in prepared]
    image_paths = [path for _, path in prepared]

    print(f"\nRunning VLM extraction → {args.out_root / args.feature}")
    extract_feature_batch(
        image_paths=image_paths,
        feature_name=args.feature,
        out_dir=str(args.out_root),
        model_name=args.model,
        image_ids=image_ids,
        skip_existing=True,
        use_few_shot=not args.no_few_shot,
        staged=args.staged,
    )
    print("\nDone. Ingest with:")
    print(
        f"  .venv/bin/python -m vision.scripts.ingest_vlm_labels --feature {args.feature}"
    )


if __name__ == "__main__":
    main()

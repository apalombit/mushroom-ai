"""Ingest per-feature VLM sidecar JSONs into image_annotations as vlm_labeled rows.

Reads sidecars produced by ``extract_feature_batch`` (one JSON per image under
``<sidecar-root>/<feature_name>/<image_id>.json``) and inserts one row per
image into ``image_annotations`` with:

- annotation_type = 'vlm_labeled'
- feature_name    = the feature being ingested
- feature_value   = sidecar's classification field; rows with a NULL value
                    (VLM abstained / cannot_tell) are skipped — the DB has a
                    NOT NULL constraint on this column and abstentions add
                    no training signal anyway
- confidence      = high → 1.0, low → 0.5, cannot_tell → 0.0
- annotator       = model name (e.g. 'gemma4:31b-cloud')

ON CONFLICT (image_id, feature_name, annotation_type) DO NOTHING — idempotent;
re-running ingests only newly added sidecars.

Usage:
    .venv/bin/python -m vision.scripts.ingest_vlm_labels \\
        --feature hymenium_type \\
        --sidecar-root data/vlm_labels \\
        --annotator gemma4:31b-cloud
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text  # noqa: E402
from tqdm import tqdm  # noqa: E402

from db.connection import get_session  # noqa: E402
from vision.labeling.vlm_feature_schemas import FEATURE_REGISTRY  # noqa: E402

_CONFIDENCE_TO_FLOAT = {"high": 1.0, "low": 0.5, "cannot_tell": 0.0}


def _classification_field(feature_name: str) -> str:
    return FEATURE_REGISTRY[feature_name]["field"]


def _iter_sidecars(feature_dir: Path):
    for path in sorted(feature_dir.glob("*.json")):
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest VLM sidecars → image_annotations")
    parser.add_argument("--feature", required=True, help="Feature name (must be in FEATURE_REGISTRY)")
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "vlm_labels",
        help="Root directory containing <feature>/<image_id>.json sidecars",
    )
    parser.add_argument(
        "--annotator",
        default="gemma4:31b-cloud",
        help="Model identifier stored in image_annotations.annotator",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="DB commit interval (default: 200 rows)",
    )
    args = parser.parse_args()

    if args.feature not in FEATURE_REGISTRY:
        raise SystemExit(
            f"Unknown feature {args.feature!r}. Known: {sorted(FEATURE_REGISTRY)}"
        )

    feature_dir = args.sidecar_root / args.feature
    if not feature_dir.exists():
        raise SystemExit(f"Sidecar dir not found: {feature_dir}")

    field = _classification_field(args.feature)
    sidecars = list(_iter_sidecars(feature_dir))
    print(f"Ingesting {len(sidecars)} sidecars from {feature_dir}")

    n_ok = 0
    n_skipped_error = 0
    n_skipped_abstain = 0
    n_inserted = 0

    with get_session() as session:
        for i, sc in enumerate(tqdm(sidecars, desc=f"ingest {args.feature}", unit="img")):
            try:
                payload = json.loads(sc.read_text())
            except Exception:
                n_skipped_error += 1
                continue

            if "error" in payload:
                n_skipped_error += 1
                continue

            value = payload.get(field)
            if value is None:
                n_skipped_abstain += 1
                continue

            confidence = _CONFIDENCE_TO_FLOAT.get(payload.get("confidence"), 0.0)
            image_id = sc.stem

            result = session.execute(
                text(
                    "INSERT INTO image_annotations "
                    "(image_id, annotation_type, feature_name, feature_value, "
                    "confidence, annotator) "
                    "VALUES (:iid, 'vlm_labeled', :fn, :fv, :cf, :ann) "
                    "ON CONFLICT (image_id, feature_name, annotation_type) DO NOTHING"
                ),
                {
                    "iid": image_id,
                    "fn": args.feature,
                    "fv": value,
                    "cf": confidence,
                    "ann": args.annotator,
                },
            )
            n_ok += 1
            n_inserted += result.rowcount or 0

            if (i + 1) % args.batch_size == 0:
                session.commit()

        session.commit()

    print(
        f"Done. parsed={n_ok} inserted={n_inserted} "
        f"skipped_error={n_skipped_error} skipped_abstain={n_skipped_abstain}"
    )


if __name__ == "__main__":
    main()

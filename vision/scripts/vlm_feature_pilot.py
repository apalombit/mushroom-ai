"""CLI: per-feature VLM extraction + three-level evaluation.

Usage examples:
    # Smoke test (5 images)
    python -m vision.scripts.vlm_feature_pilot --feature hymenium_type --n 5

    # Full test split extraction + eval
    python -m vision.scripts.vlm_feature_pilot --feature hymenium_type --split test

    # Compare against all-at-once VLM baseline
    python -m vision.scripts.vlm_feature_pilot --feature cap_color --split test --compare-all

    # Resume (skip already-extracted sidecars)
    python -m vision.scripts.vlm_feature_pilot --feature hymenium_type --split test --resume
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from db.connection import get_session  # noqa: E402
from vision.data.ground_truth import load_ground_truth  # noqa: E402
from vision.labeling.vlm_feature_extractor import extract_feature_batch  # noqa: E402
from vision.labeling.vlm_feature_schemas import FEATURE_REGISTRY  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "images" / "processed" / "v1"
VLM_FULL_DIR = PROJECT_ROOT / "data" / "vlm_full" / "run_001"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "vlm_features"

# Map feature → field name in the all-at-once VLM sidecar
_VLM_FULL_FIELD = {
    "hymenium_type": "hymenium_type",
    "cap_color": "cap_color_primary",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_manifest(session, feature_name: str, split: str | None, n: int | None):
    """Load image manifest from DB, optionally filtered to a split.

    Uses the same quality/visibility SQL filter as build_manifest but checks
    for processed .jpg files instead of .pt embeddings.

    Returns DataFrame with columns: image_id, species, feature_value, processed_path.
    """
    from vision.data.manifest import _build_filtered_query  # noqa: E402
    from vision.data.splits import stratified_species_split  # noqa: E402

    query, params = _build_filtered_query(feature_name)
    rows = session.execute(query, params).fetchall()

    if not rows:
        return pd.DataFrame(columns=["image_id", "species", "feature_value", "processed_path"])

    df = pd.DataFrame(rows, columns=["image_id", "species", "feature_value"])

    # Filter to images with processed files on disk
    df["processed_path"] = df["image_id"].apply(lambda iid: str(PROCESSED_DIR / f"{iid}.jpg"))
    df = df[df["processed_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    print(
        f"  manifest: {len(df)} images, "
        f"{df['feature_value'].nunique()} classes, "
        f"{df['species'].nunique()} species"
    )

    if split:
        train_df, val_df, test_df = stratified_species_split(df)
        split_map = {"train": train_df, "val": val_df, "test": test_df}
        if split not in split_map:
            print(f"ERROR: unknown split '{split}'. Use train/val/test.")
            sys.exit(1)
        df = split_map[split]

    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=42).reset_index(drop=True)

    return df


def _load_vlm_full_predictions(image_ids: list[str], feature_name: str) -> dict:
    """Load all-at-once VLM predictions from existing sidecars."""
    field = _VLM_FULL_FIELD.get(feature_name)
    if not field:
        return {}
    preds = {}
    for iid in image_ids:
        sidecar = VLM_FULL_DIR / f"{iid}.json"
        if not sidecar.exists():
            continue
        try:
            data = json.loads(sidecar.read_text())
            preds[iid] = data.get(field)
        except (json.JSONDecodeError, KeyError):
            continue
    return preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _per_image_accuracy(
    preds: dict[str, str | None],
    labels: dict[str, str | None],
    name: str = "per-feature VLM",
) -> dict:
    """L1: per-image accuracy (excluding cannot_tell/null predictions)."""
    correct = 0
    total = 0
    conf_counts: Counter = Counter()

    for iid, pred in preds.items():
        label = labels.get(iid)
        if label is None:
            continue
        if pred is None:
            conf_counts["null"] += 1
            continue
        total += 1
        if pred == label:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    print(f"\n  [{name}] L1 per-image accuracy: {acc:.1%} ({correct}/{total})")
    print(f"    null/cannot_tell predictions: {conf_counts['null']}")
    return {"accuracy": acc, "correct": correct, "total": total, "null": conf_counts["null"]}


def _species_consensus(
    image_df: pd.DataFrame,
    preds: dict[str, str | None],
    gt: dict[str, str | None],
    feature_field: str,
    name: str = "per-feature VLM",
) -> dict:
    """L2: species-level majority-vote consensus vs ground truth."""
    # Group predictions by species
    species_preds: dict[str, list[str]] = {}
    for _, row in image_df.iterrows():
        iid = row["image_id"]
        sp = row["species"]
        pred = preds.get(iid)
        if pred is not None:
            species_preds.setdefault(sp, []).append(pred)

    correct = 0
    total = 0
    for sp, pred_list in species_preds.items():
        truth = gt.get(sp)
        if truth is None:
            continue
        # Majority vote
        consensus = Counter(pred_list).most_common(1)[0][0]
        total += 1
        if consensus == truth:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    print(f"  [{name}] L2 species consensus: {acc:.1%} ({correct}/{total} species)")
    return {"accuracy": acc, "correct": correct, "total": total}


def _confidence_breakdown(results: list[dict], feature_field: str):
    """Print confidence distribution."""
    conf_counts: Counter = Counter()

    for r in results:
        if r["result"] is None:
            continue
        conf = r["result"].get("confidence", "unknown")
        conf_counts[conf] += 1

    print("\n  Confidence distribution:")
    for level in ["high", "low", "cannot_tell"]:
        print(f"    {level}: {conf_counts.get(level, 0)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Per-feature VLM extraction + evaluation")
    parser.add_argument(
        "--feature",
        required=True,
        choices=sorted(FEATURE_REGISTRY),
        help="Feature to extract",
    )
    parser.add_argument("--n", type=int, default=None, help="Limit to N images (random sample)")
    parser.add_argument("--split", type=str, default=None, help="Use train/val/test split")
    parser.add_argument("--model", type=str, default="gemma4:31b-cloud", help="VLM model name")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Skip existing sidecars")
    parser.add_argument(
        "--compare-all", action="store_true", help="Include all-at-once VLM baseline"
    )
    parser.add_argument("--eval-only", action="store_true", help="Skip extraction, eval existing")
    args = parser.parse_args()

    feature = args.feature
    feature_field = FEATURE_REGISTRY[feature]["field"]
    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR

    print(f"Feature: {feature}")
    print(f"Output:  {out_dir / feature}")

    # Load manifest from DB
    print("\nLoading manifest from DB...")
    with get_session() as session:
        df = _load_manifest(session, feature, args.split, args.n)
        gt = load_ground_truth(session, feature)

    if df.empty:
        print("ERROR: no images in manifest.")
        sys.exit(1)

    print(f"  images: {len(df)}, species: {df['species'].nunique()}")

    # ── Extraction ────────────────────────────────────────────────────────
    if not args.eval_only:
        print(f"\nExtracting {feature} with {args.model}...")
        results = extract_feature_batch(
            image_paths=df["processed_path"].tolist(),
            feature_name=feature,
            out_dir=out_dir,
            model_name=args.model,
            image_ids=df["image_id"].tolist(),
            skip_existing=args.resume,
        )

        n_ok = sum(1 for r in results if r["result"] is not None)
        n_err = sum(1 for r in results if r["error"] is not None)
        n_skip = sum(1 for r in results if r["skipped"])
        print(f"\n  extracted: {n_ok}, errors: {n_err}, skipped: {n_skip}")
    else:
        results = None

    # ── Load predictions from sidecars ────────────────────────────────────
    feature_dir = out_dir / feature
    preds: dict[str, str | None] = {}
    result_list: list[dict] = []

    for _, row in df.iterrows():
        iid = row["image_id"]
        sidecar = feature_dir / f"{iid}.json"
        if not sidecar.exists():
            continue
        try:
            data = json.loads(sidecar.read_text())
            if "error" in data and data.get("error") is not None:
                continue
            preds[iid] = data.get(feature_field)
            result_list.append({"image_id": iid, "result": data})
        except json.JSONDecodeError:
            continue

    print(f"\n  loaded {len(preds)} predictions from sidecars")

    # ── Evaluation ─────────────────────────────────────────────────────────
    # Build per-image ground truth from species-level GT
    image_labels = {}
    for _, row in df.iterrows():
        sp_gt = gt.get(row["species"])
        if sp_gt is not None:
            image_labels[row["image_id"]] = sp_gt

    print(f"  images with ground truth: {len(image_labels)}")
    print("\n── Evaluation ──")

    # L1: per-image accuracy
    _per_image_accuracy(preds, image_labels, name="per-feature VLM")

    # L2: species consensus
    _species_consensus(df, preds, gt, feature_field, name="per-feature VLM")

    # Confidence breakdown
    if result_list:
        _confidence_breakdown(result_list, feature_field)

    # ── Baseline: all-at-once VLM ──────────────────────────────────────────
    if args.compare_all:
        print("\n── All-at-once VLM baseline ──")
        vlm_full = _load_vlm_full_predictions(df["image_id"].tolist(), feature)
        print(f"  loaded {len(vlm_full)} all-at-once predictions")

        # For cap_color, all-at-once uses free text → need alias resolution
        if feature == "cap_color":
            from ingestion.normalize import load_vocabulary  # noqa: E402

            vocab = load_vocabulary()
            color_aliases = vocab.get("color_palette", {}).get("aliases", {})
            # Simple alias resolution for free-text cap_color_primary
            resolved = {}
            from vision.labeling.vlm_feature_schemas import _CLASSES  # noqa: E402

            canonical_colors = set(_CLASSES["cap_color"])
            for iid, val in vlm_full.items():
                if val is None:
                    resolved[iid] = None
                elif val.lower() in canonical_colors:
                    resolved[iid] = val.lower()
                elif val.lower() in color_aliases:
                    resolved[iid] = color_aliases[val.lower()]
                else:
                    resolved[iid] = None  # unmappable free text
            vlm_full = resolved

        _per_image_accuracy(vlm_full, image_labels, name="all-at-once VLM")
        _species_consensus(df, vlm_full, gt, feature_field, name="all-at-once VLM")

    print("\nDone.")


if __name__ == "__main__":
    main()

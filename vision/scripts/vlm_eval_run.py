"""Species-stratified VLM evaluation runner.

Supports dual-feature mode (both hymenium_type + cap_color, joint intersection)
or single-feature mode (--feature cap_color) which draws from the full pool for
that feature alone, giving much larger eval sets.

Usage examples:
    # Dual-feature (joint intersection, smaller pool)
    python -m vision.scripts.vlm_eval_run --n 300 --model gemma4:31b-cloud

    # Single-feature (full pool, bigger eval set)
    python -m vision.scripts.vlm_eval_run --feature cap_color --n 800 --model gemma4:31b-cloud

    # Preview selection only (no VLM calls)
    python -m vision.scripts.vlm_eval_run --feature cap_color --n 800 --selection-only

    # Resume interrupted run
    python -m vision.scripts.vlm_eval_run --feature cap_color --n 800 --resume
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from db.connection import get_session  # noqa: E402
from vision.data.manifest import _build_filtered_query  # noqa: E402
from vision.data.splits import stratified_species_split  # noqa: E402
from vision.labeling.vlm_feature_extractor import extract_feature_batch  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "images" / "processed" / "v1"
RAW_DIR = PROJECT_ROOT / "data" / "images" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "vlm_eval"

ALL_FEATURES = ["hymenium_type", "cap_color", "ring_presence", "volva_presence"]


def _center_crop_square(img):
    """Center-crop to square using the shorter side."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _prepare_vlm_images(
    eval_df: pd.DataFrame,
    vlm_resolution: int | None,
    square_strategy: str | None = None,
) -> list[str]:
    """Return image paths for VLM extraction, optionally preprocessed.

    Modes:
    - vlm_resolution=None, square_strategy=None: return processed/v1 paths as-is (256px).
    - vlm_resolution=N, square_strategy=None: shortest-side resize to N (legacy).
    - vlm_resolution=N, square_strategy="center_crop": center-crop to square at native
      resolution, then resize to NxN. Avoids upscaling (when raw min-side >= N) and
      preserves the central subject's true aspect ratio. Recommended for Gemma 3
      (encoder native = 896x896 square; staying at/under 896 avoids Pan&Scan).
    """
    if vlm_resolution is None and square_strategy is None:
        return eval_df["processed_path"].tolist()

    from PIL import Image

    if square_strategy == "center_crop":
        if vlm_resolution is None:
            raise ValueError("square_strategy='center_crop' requires vlm_resolution")
        cache_name = f"vlm_{vlm_resolution}_cc"
    elif square_strategy is None:
        cache_name = f"vlm_{vlm_resolution}"
    else:
        raise ValueError(f"unknown square_strategy: {square_strategy!r}")

    cache_dir = PROJECT_ROOT / "data" / "images" / "processed" / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    written = 0
    for _, row in eval_df.iterrows():
        iid = row["image_id"]
        cached = cache_dir / f"{iid}.jpg"
        if cached.exists():
            paths.append(str(cached))
            continue

        raw = RAW_DIR / f"{iid}.jpg"
        if not raw.exists():
            paths.append(row["processed_path"])
            continue

        img = Image.open(raw).convert("RGB")

        if square_strategy == "center_crop":
            img = _center_crop_square(img)
            img = img.resize((vlm_resolution, vlm_resolution), Image.LANCZOS)
        else:
            w, h = img.size
            short_side = min(w, h)
            if short_side > vlm_resolution:
                scale = vlm_resolution / short_side
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        img.save(cached, "JPEG", quality=95)
        paths.append(str(cached))
        written += 1

    suffix = f" ({square_strategy})" if square_strategy else ""
    print(
        f"  VLM resolution: {vlm_resolution}px{suffix} "
        f"(prepared {written} images, cache: {cache_dir})"
    )
    return paths


# ---------------------------------------------------------------------------
# Single-feature selection
# ---------------------------------------------------------------------------


def select_single_feature_images(
    session,
    feature_name: str,
    target_n: int = 800,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select eval images for a single feature from its full pool.

    Algorithm:
        1. Load manifest for the feature (no joint intersection).
        2. Filter to images with processed files on disk.
        3. Stratified species split → test set.
        4. Class-aware species selection targeting ~target_n images.

    Returns:
        DataFrame with: image_id, species, {feature_name}, processed_path.
    """
    query, params = _build_filtered_query(feature_name)
    rows = session.execute(query, params).fetchall()
    df = pd.DataFrame(rows, columns=["image_id", "species", feature_name])

    df["processed_path"] = df["image_id"].apply(
        lambda iid: str(PROCESSED_DIR / f"{iid}.jpg")
    )
    df = df[df["processed_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    print(f"{feature_name} pool: {len(df)} images, {df['species'].nunique()} species")
    print(f"  classes: {df[feature_name].nunique()} ({', '.join(df[feature_name].unique())})")

    if df.empty:
        return df

    # Stratified split — use larger test ratio when pool is big enough
    test_ratio = 0.25 if len(df) > 1000 else 0.15
    train_ratio = 1.0 - test_ratio - 0.15
    split_df = df.copy()
    split_df["feature_value"] = df[feature_name]
    _, _, test_df = stratified_species_split(
        split_df, train_ratio=train_ratio, val_ratio=0.15,
        test_ratio=test_ratio, random_state=random_state,
    )
    test_df = test_df.drop(columns=["feature_value", "split"], errors="ignore")

    print(f"Test split: {len(test_df)} images, {test_df['species'].nunique()} species")

    # Class-aware selection
    selected = _single_feature_select(test_df, feature_name, target_n, random_state)
    eval_df = test_df[test_df["species"].isin(selected)].reset_index(drop=True)

    print(
        f"Selected: {len(eval_df)} images, "
        f"{eval_df['species'].nunique()} species "
        f"(target was {target_n})"
    )
    return eval_df


def _single_feature_select(
    test_df: pd.DataFrame,
    feature_name: str,
    target_n: int,
    random_state: int,
) -> set[str]:
    """Class-aware species selection for a single feature.

    Pass 1: For each class (rarest first), pick ≤3 species to ensure coverage.
    Pass 2: Random fill until image count ≈ target_n.
    """
    sp_stats = (
        test_df.groupby("species")
        .agg(
            n_images=("image_id", "count"),
            class_mode=(feature_name, lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )

    selected: set[str] = set()
    image_count = 0
    max_images = int(target_n * 1.15)

    def _add(sp_name: str):
        nonlocal image_count
        if sp_name in selected:
            return
        n = sp_stats.loc[sp_stats["species"] == sp_name, "n_images"].iloc[0]
        selected.add(sp_name)
        image_count += n

    # Pass 1: cover every class, rarest first, ≤3 species each
    class_counts = test_df[feature_name].value_counts()
    for cls in class_counts.index[::-1]:
        candidates = sp_stats[
            (sp_stats["class_mode"] == cls) & (~sp_stats["species"].isin(selected))
        ]
        if candidates.empty:
            continue
        picks = candidates.sample(n=min(3, len(candidates)), random_state=random_state)
        for sp in picks["species"]:
            _add(sp)

    # Pass 2: random fill
    remaining = sp_stats[~sp_stats["species"].isin(selected)].copy()
    remaining = remaining.sample(frac=1, random_state=random_state)

    for _, row in remaining.iterrows():
        if image_count >= max_images:
            break
        if image_count + row["n_images"] > max_images and image_count >= target_n * 0.85:
            continue
        _add(row["species"])

    return selected


# ---------------------------------------------------------------------------
# Dual-feature selection (original)
# ---------------------------------------------------------------------------


def select_eval_images(
    session,
    target_n: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """Species-stratified selection from joint test split (both features)."""
    dfs = {}
    for feat in ALL_FEATURES:
        query, params = _build_filtered_query(feat)
        rows = session.execute(query, params).fetchall()
        df = pd.DataFrame(rows, columns=["image_id", "species", "feature_value"])
        df = df.rename(columns={"feature_value": feat})
        dfs[feat] = df

    merged = dfs["hymenium_type"].merge(dfs["cap_color"], on=["image_id", "species"], how="inner")
    merged["processed_path"] = merged["image_id"].apply(
        lambda iid: str(PROCESSED_DIR / f"{iid}.jpg")
    )
    merged = merged[merged["processed_path"].apply(lambda p: Path(p).exists())].reset_index(
        drop=True
    )

    print(f"Joint intersection: {len(merged)} images, {merged['species'].nunique()} species")
    if merged.empty:
        return merged

    split_df = merged.copy()
    split_df["feature_value"] = merged["hymenium_type"]
    _, _, test_df = stratified_species_split(split_df, random_state=random_state)
    test_df = test_df.drop(columns=["feature_value", "split"], errors="ignore")
    print(f"Test split: {len(test_df)} images, {test_df['species'].nunique()} species")

    selected_species = _dual_feature_select(test_df, target_n, random_state)
    eval_df = test_df[test_df["species"].isin(selected_species)].reset_index(drop=True)
    print(
        f"Selected: {len(eval_df)} images, "
        f"{eval_df['species'].nunique()} species "
        f"(target was {target_n})"
    )
    return eval_df


def _dual_feature_select(
    test_df: pd.DataFrame,
    target_n: int,
    random_state: int,
) -> set[str]:
    """Three-pass class-aware species selection for dual-feature mode."""
    sp_stats = (
        test_df.groupby("species")
        .agg(
            n_images=("image_id", "count"),
            hymenium_mode=("hymenium_type", lambda x: x.mode().iloc[0]),
            cap_color_mode=("cap_color", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )

    selected: set[str] = set()
    image_count = 0
    max_images = int(target_n * 1.15)

    def _add(sp_name: str):
        nonlocal image_count
        if sp_name in selected:
            return
        n = sp_stats.loc[sp_stats["species"] == sp_name, "n_images"].iloc[0]
        selected.add(sp_name)
        image_count += n

    # Pass 1: hymenium classes
    hy_counts = test_df["hymenium_type"].value_counts()
    for cls in hy_counts.index[::-1]:
        candidates = sp_stats[
            (sp_stats["hymenium_mode"] == cls) & (~sp_stats["species"].isin(selected))
        ]
        if candidates.empty:
            continue
        picks = candidates.sample(n=min(2, len(candidates)), random_state=random_state)
        for sp in picks["species"]:
            _add(sp)

    # Pass 2: cap_color classes
    covered_colors = {
        sp_stats.loc[sp_stats["species"] == sp, "cap_color_mode"].iloc[0] for sp in selected
    }
    cc_counts = test_df["cap_color"].value_counts()
    for cls in cc_counts.index[::-1]:
        if cls in covered_colors:
            continue
        candidates = sp_stats[
            (sp_stats["cap_color_mode"] == cls) & (~sp_stats["species"].isin(selected))
        ]
        if candidates.empty:
            continue
        pick = candidates.sample(n=1, random_state=random_state)
        _add(pick["species"].iloc[0])
        covered_colors.add(cls)

    # Pass 3: random fill
    remaining = sp_stats[~sp_stats["species"].isin(selected)].copy()
    remaining = remaining.sample(frac=1, random_state=random_state)
    for _, row in remaining.iterrows():
        if image_count >= max_images:
            break
        if image_count + row["n_images"] > max_images and image_count >= target_n * 0.85:
            continue
        _add(row["species"])

    return selected


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def _print_class_distribution(df: pd.DataFrame, features: list[str]):
    """Print class distributions for the given features."""
    for feat in features:
        if feat not in df.columns:
            continue
        counts = df[feat].value_counts()
        n_species = df.groupby(feat)["species"].nunique()
        print(f"\n  {feat} distribution:")
        for cls in counts.index:
            print(f"    {cls:20s}  {counts[cls]:4d} imgs  {n_species[cls]:3d} spp")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Species-stratified VLM evaluation")
    parser.add_argument(
        "--n", type=int, default=300, help="Target number of images (default: 300)"
    )
    parser.add_argument(
        "--feature", type=str, default=None,
        help="Single feature to evaluate (e.g. cap_color). Omit for dual-feature mode.",
    )
    parser.add_argument("--model", type=str, default="gemma4:31b-cloud", help="VLM model name")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Skip existing sidecars")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Just save manifest + stats, no VLM calls",
    )
    parser.add_argument(
        "--vlm-resolution",
        type=int,
        default=None,
        help="Resize raw images to this shortest-side px for VLM (default: 256px processed)",
    )
    parser.add_argument(
        "--square",
        choices=["center_crop"],
        default=None,
        help=(
            "Square preprocessing strategy. With center_crop, --vlm-resolution sets "
            "the target side of a square output (e.g. 896 for Gemma 3 native)."
        ),
    )
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot reference images even if reference_images/ exists.",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help=(
            "Use two-stage chain-of-inquiry extraction (Path C). Stage 1 picks "
            "a coarse family, stage 2 disambiguates within the family. Implies "
            "--no-few-shot for now (staged + few-shot is not implemented)."
        ),
    )
    args = parser.parse_args()

    if args.staged and not args.no_few_shot:
        # Staged mode is a separate code path; we don't currently support
        # combining it with few-shot reference images.
        print(
            "  --staged implies no few-shot (turning few-shot off for this run)."
        )

    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    single_feature = args.feature
    if single_feature and single_feature not in ALL_FEATURES:
        print(f"ERROR: unknown feature '{single_feature}'. Choose from {ALL_FEATURES}")
        sys.exit(1)

    # ── Selection ─────────────────────────────────────────────────────────
    print("Loading manifests from DB...")
    with get_session() as session:
        if single_feature:
            eval_df = select_single_feature_images(
                session, single_feature, target_n=args.n
            )
            features = [single_feature]
        else:
            eval_df = select_eval_images(session, target_n=args.n)
            features = ALL_FEATURES

    if eval_df.empty:
        print("ERROR: no eligible images found.")
        sys.exit(1)

    _print_class_distribution(eval_df, features)

    # Save manifest
    manifest_path = out_dir / "eval_manifest.csv"
    eval_df.to_csv(manifest_path, index=False)
    print(f"\nManifest saved: {manifest_path} ({len(eval_df)} images)")

    if args.selection_only:
        print("\n--selection-only: stopping before extraction.")
        return

    # ── Extraction ────────────────────────────────────────────────────────
    vlm_paths = _prepare_vlm_images(eval_df, args.vlm_resolution, args.square)

    for feat in features:
        print(f"\n{'=' * 60}")
        print(f"Extracting {feat} with {args.model}...")
        print(f"{'=' * 60}")

        results = extract_feature_batch(
            image_paths=vlm_paths,
            feature_name=feat,
            out_dir=out_dir,
            model_name=args.model,
            image_ids=eval_df["image_id"].tolist(),
            skip_existing=args.resume,
            use_few_shot=(not args.no_few_shot) and (not args.staged),
            staged=args.staged,
        )

        n_ok = sum(1 for r in results if r["result"] is not None)
        n_err = sum(1 for r in results if r["error"] is not None)
        n_skip = sum(1 for r in results if r["skipped"])
        print(f"\n  {feat}: extracted={n_ok}, errors={n_err}, skipped={n_skip}")

    print("\nDone. Run the report notebook to analyze results.")


if __name__ == "__main__":
    main()

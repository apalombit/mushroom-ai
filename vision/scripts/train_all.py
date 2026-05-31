"""CLI: train all active heads in a specified tier sequentially."""

import argparse
from pathlib import Path

import yaml

from vision.training.trainer import train_head

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def main():
    parser = argparse.ArgumentParser(description="Train all vision heads for a tier")
    parser.add_argument("--tier", type=int, required=True, help="Tier number (1, 2, ...)")
    parser.add_argument("--preprocess-version", default="v1")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable VLM quality filtering (use all images)",
    )
    args = parser.parse_args()

    with open(CONFIG_DIR / "features.yaml") as f:
        features = yaml.safe_load(f)

    tier_features = [name for name, cfg in features.items() if cfg.get("tier") == args.tier]

    if not tier_features:
        print(f"No features found for tier {args.tier}")
        return

    print(f"Training tier {args.tier} heads: {', '.join(tier_features)}\n")

    results = []
    for feature_name in tier_features:
        print(f"\n{'=' * 60}")
        print(f"  {feature_name}")
        print(f"{'=' * 60}")
        result = train_head(
            feature_name=feature_name,
            preprocess_version=args.preprocess_version,
            device=args.device,
            quality_filter=not args.no_quality_filter,
        )
        results.append(result)

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — Tier {args.tier}")
    print(f"{'=' * 60}")
    for r in results:
        print(
            f"  {r.feature_name:25s}  "
            f"val_f1={r.best_val_f1:.3f}  "
            f"test_acc={r.test_metrics['accuracy']:.3f}  "
            f"test_f1={r.test_metrics['macro_f1']:.3f}  "
            f"(epoch {r.best_epoch})"
        )


if __name__ == "__main__":
    main()

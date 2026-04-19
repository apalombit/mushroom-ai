"""CLI: train a single feature head on cached embeddings."""

import argparse

from vision.training.trainer import train_head


def main():
    parser = argparse.ArgumentParser(description="Train a single vision feature head")
    parser.add_argument("--feature", required=True, help="Feature name (e.g. hymenium_type)")
    parser.add_argument("--preprocess-version", default="v1")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable VLM quality filtering (use all images)",
    )
    args = parser.parse_args()

    train_head(
        feature_name=args.feature,
        preprocess_version=args.preprocess_version,
        lr=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        dropout=args.dropout,
        batch_size=args.batch_size,
        device=args.device,
        quality_filter=not args.no_quality_filter,
    )


if __name__ == "__main__":
    main()

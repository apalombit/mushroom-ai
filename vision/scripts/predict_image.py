"""CLI: run inference on image(s), print predictions with confidence."""

import argparse
from pathlib import Path

from vision.inference.predict import predict_image
from vision.models.embed import EmbeddingExtractor
from vision.models.multi_head import MultiHeadPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict morphological features from image")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--checkpoint-dir", default="data/checkpoints")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--heads",
        default=None,
        help="Comma-separated head names to run (default: all available)",
    )
    args = parser.parse_args()

    predictor = MultiHeadPredictor(args.checkpoint_dir, device=args.device)

    if not predictor.feature_names:
        print(f"No trained heads found in {args.checkpoint_dir}")
        return

    if args.heads:
        requested = set(args.heads.split(","))
        missing = requested - set(predictor.feature_names)
        if missing:
            print(f"Heads not found: {missing}")
            return

    extractor = EmbeddingExtractor(device=args.device)
    results = predict_image(args.image, predictor, extractor)

    if args.heads:
        requested = set(args.heads.split(","))
        results = {k: v for k, v in results.items() if k in requested}

    image_name = Path(args.image).name
    print(f"\nPredictions for: {image_name}")
    print("-" * 60)
    for feat, pred in results.items():
        status = pred["status"]
        value = pred["value"] or "-"
        conf = pred["confidence"]
        runner = pred["runner_up"]
        runner_conf = pred["runner_up_confidence"]
        print(
            f"  {feat:25s}  {value:20s}  ({conf:.2f})  "
            f"[{status}]  runner-up: {runner} ({runner_conf:.2f})"
        )


if __name__ == "__main__":
    main()

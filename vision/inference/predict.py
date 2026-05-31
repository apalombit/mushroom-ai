"""Single-image prediction: preprocess -> embed -> run all active heads."""

from pathlib import Path

import torch

from vision.models.embed import EmbeddingExtractor
from vision.models.multi_head import MultiHeadPredictor


def predict_from_embedding(
    embedding: torch.Tensor, predictor: MultiHeadPredictor
) -> dict[str, dict]:
    """Run all heads on a precomputed embedding."""
    return predictor.predict(embedding)


def predict_image(
    image_path: str | Path,
    predictor: MultiHeadPredictor,
    extractor: EmbeddingExtractor,
) -> dict[str, dict]:
    """End-to-end prediction: load image -> embed -> predict all heads.

    Args:
        image_path: Path to an image file (raw or processed).
        predictor: MultiHeadPredictor with loaded heads.
        extractor: EmbeddingExtractor for DINOv2 embedding.

    Returns:
        {feature_name: prediction_dict} using universal output contract.
    """
    embedding = extractor.extract_single(Path(image_path))
    return predictor.predict(embedding)

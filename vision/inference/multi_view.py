"""Aggregate predictions across multiple views of the same specimen."""

from pathlib import Path

import torch
import torch.nn.functional as F

from vision.models.embed import EmbeddingExtractor
from vision.models.multi_head import MultiHeadPredictor


def predict_specimen(
    image_paths: list[str | Path],
    predictor: MultiHeadPredictor,
    extractor: EmbeddingExtractor,
) -> dict[str, dict]:
    """Predict features by averaging softmax across multiple images.

    For each head, computes softmax per image then averages the distributions
    before picking the argmax. This is more robust than single-image prediction.

    Args:
        image_paths: List of image paths for the same specimen.
        predictor: MultiHeadPredictor with loaded heads.
        extractor: EmbeddingExtractor for DINOv2 embeddings.

    Returns:
        {feature_name: prediction_dict} with universal output contract,
        plus 'n_images' key indicating how many images contributed.
    """
    # Extract embeddings for all images
    embeddings = []
    for path in image_paths:
        emb = extractor.extract_single(Path(path))
        embeddings.append(emb)

    if not embeddings:
        return {}

    device = predictor.device
    results = {}

    for feat_name, head in predictor.heads.items():
        head.eval()

        # Collect softmax distributions across all images
        all_probs = []
        with torch.no_grad():
            for emb in embeddings:
                emb_d = emb.unsqueeze(0).to(device)
                logits = head(emb_d)
                probs = F.softmax(logits / head.temperature, dim=-1).squeeze(0)
                all_probs.append(probs)

        # Average softmax distributions
        avg_probs = torch.stack(all_probs).mean(dim=0)

        top2 = torch.topk(avg_probs, k=min(2, len(avg_probs)))
        top_prob = top2.values[0].item()
        top_idx = top2.indices[0].item()
        runner_up_prob = top2.values[1].item() if len(top2.values) > 1 else 0.0
        runner_up_idx = top2.indices[1].item() if len(top2.indices) > 1 else 0

        margin = top_prob - runner_up_prob

        if top_prob >= 0.6 and margin >= 0.2:
            status = "confident"
        elif top_prob >= 0.3:
            status = "uncertain"
        else:
            status = "not_determinable"

        results[feat_name] = {
            "status": status,
            "value": head.class_names[top_idx] if status != "not_determinable" else None,
            "confidence": top_prob,
            "runner_up": head.class_names[runner_up_idx],
            "runner_up_confidence": runner_up_prob,
            "n_images": len(embeddings),
        }

    return results

"""CrossEntropyLoss with class weights and label smoothing."""

import torch
import torch.nn as nn

from vision.data.dataset import MushroomFeatureDataset


def compute_class_weights(dataset: MushroomFeatureDataset) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to sum to num_classes."""
    counts = dataset.class_counts
    n = len(dataset)
    num_classes = dataset.num_classes
    weights = []
    for cls_name in dataset.class_names:
        c = counts[cls_name]
        weights.append(n / (num_classes * c) if c > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def build_loss(
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.1,
    device: str = "cpu",
) -> nn.CrossEntropyLoss:
    """Build CrossEntropyLoss with optional class weights and label smoothing."""
    weight = class_weights.to(device) if class_weights is not None else None
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

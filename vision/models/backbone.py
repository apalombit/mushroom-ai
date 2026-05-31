"""Frozen DINOv2-B/14 feature extractor. Returns CLS token (768-d)."""

import torch
from transformers import AutoModel


class FeatureExtractor:
    """Wraps facebook/dinov2-base for frozen feature extraction."""

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = "mps"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def embed_dim(self) -> int:
        return self.model.config.hidden_size  # 768 for dinov2-base

    @torch.no_grad()
    def extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract CLS token embeddings.

        Args:
            pixel_values: (B, 3, 224, 224) float tensor, ImageNet-normalized.

        Returns:
            (B, 768) tensor of CLS token features.
        """
        pixel_values = pixel_values.to(self.device)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

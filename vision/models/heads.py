"""FeatureHead: MLP probe per morphological feature (Linear -> BN -> ReLU -> Linear)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

_HIDDEN_DIM1 = 256
_HIDDEN_DIM2 = 128


class FeatureHead(nn.Module):
    """Two-layer MLP classification head.

    Linear(embed_dim, 256) -> BN -> ReLU -> Dropout
    -> Linear(256, 128) -> BN -> ReLU -> Dropout
    -> Linear(128, num_classes)

    Stores metadata (feature_name, class_names, temperature) alongside weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        feature_name: str,
        class_names: list[str],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_name = feature_name
        self.class_names = class_names
        self.temperature = 1.0
        self.hidden = nn.Linear(embed_dim, _HIDDEN_DIM1)
        self.bn = nn.BatchNorm1d(_HIDDEN_DIM1)
        self.hidden2 = nn.Linear(_HIDDEN_DIM1, _HIDDEN_DIM2)
        self.bn2 = nn.BatchNorm1d(_HIDDEN_DIM2)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(_HIDDEN_DIM2, num_classes)

    @property
    def num_classes(self) -> int:
        return self.linear.out_features

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Raw logits. Shape: (B, num_classes)."""
        x = self.drop(F.relu(self.bn(self.hidden(embedding))))
        x = self.drop(F.relu(self.bn2(self.hidden2(x))))
        return self.linear(x)

    def predict(self, embedding: torch.Tensor) -> dict:
        """Single-sample prediction with universal output contract.

        Args:
            embedding: (768,) or (1, 768) tensor.

        Returns:
            dict with: status, value, confidence, runner_up, runner_up_confidence.
        """
        self.eval()
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(embedding)
            probs = F.softmax(logits / self.temperature, dim=-1).squeeze(0)

        top2 = torch.topk(probs, k=min(2, len(probs)))
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

        return {
            "status": status,
            "value": self.class_names[top_idx] if status != "not_determinable" else None,
            "confidence": top_prob,
            "runner_up": self.class_names[runner_up_idx],
            "runner_up_confidence": runner_up_prob,
        }

    def save(self, path: str):
        """Save head with metadata."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "feature_name": self.feature_name,
                "class_names": self.class_names,
                "temperature": self.temperature,
                "embed_dim": self.hidden.in_features,
                "num_classes": self.num_classes,
                "hidden_dim": _HIDDEN_DIM1,
                "hidden_dim2": _HIDDEN_DIM2,
                "dropout": self.drop.p,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "FeatureHead":
        """Load head from checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        head = cls(
            embed_dim=ckpt["embed_dim"],
            num_classes=ckpt["num_classes"],
            feature_name=ckpt["feature_name"],
            class_names=ckpt["class_names"],
            dropout=ckpt.get("dropout", 0.1),
        )
        head.load_state_dict(ckpt["state_dict"])
        head.temperature = ckpt.get("temperature", 1.0)
        head.to(device)
        head.eval()
        return head

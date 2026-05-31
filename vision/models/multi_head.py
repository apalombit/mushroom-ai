"""MultiHeadPredictor: wrapper running multiple FeatureHeads on one embedding."""

from pathlib import Path

import torch

from vision.models.heads import FeatureHead


class MultiHeadPredictor:
    """Loads all saved heads from a checkpoint directory and runs them on one embedding."""

    def __init__(self, checkpoint_dir: str | Path = "data/checkpoints", device: str = "cpu"):
        self.device = device
        self.heads: dict[str, FeatureHead] = {}
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            return
        for head_dir in sorted(ckpt_dir.iterdir()):
            best = head_dir / "best.pt"
            if best.exists():
                head = FeatureHead.load(str(best), device=device)
                self.heads[head.feature_name] = head

    @property
    def feature_names(self) -> list[str]:
        return list(self.heads.keys())

    def predict(self, embedding: torch.Tensor) -> dict[str, dict]:
        """Run all heads on a single embedding.

        Args:
            embedding: (768,) tensor.

        Returns:
            {feature_name: prediction_dict} using universal output contract.
        """
        embedding = embedding.to(self.device)
        return {name: head.predict(embedding) for name, head in self.heads.items()}

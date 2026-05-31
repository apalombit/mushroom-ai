"""MushroomFeatureDataset: PyTorch Dataset loading cached embeddings + labels."""

import pandas as pd
import torch
from torch.utils.data import Dataset


class MushroomFeatureDataset(Dataset):
    """Dataset that loads cached DINOv2 embeddings and returns (embedding, label_idx).

    No image loading — reads .pt tensors from disk.
    """

    def __init__(self, manifest_df: pd.DataFrame, class_names: list[str]):
        """
        Args:
            manifest_df: DataFrame with columns: image_id, feature_value, embedding_path.
            class_names: Ordered list of canonical class names (index = label).
        """
        self._class_names = class_names
        self._class_to_idx = {c: i for i, c in enumerate(class_names)}

        # Filter to rows with valid class values
        valid = manifest_df[manifest_df["feature_value"].isin(self._class_to_idx)]
        if len(valid) < len(manifest_df):
            dropped = len(manifest_df) - len(valid)
            print(f"  Dropped {dropped} rows with unknown feature values")

        self._image_ids = valid["image_id"].tolist()
        self._paths = valid["embedding_path"].tolist()
        self._labels = [self._class_to_idx[v] for v in valid["feature_value"]]

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        embedding = torch.load(self._paths[idx], weights_only=True)
        return embedding, self._labels[idx], self._image_ids[idx]

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def class_to_idx(self) -> dict[str, int]:
        return self._class_to_idx

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    @property
    def class_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in self._class_names}
        for label_idx in self._labels:
            counts[self._class_names[label_idx]] += 1
        return counts

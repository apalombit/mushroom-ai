"""Stratified train/val/test splitting by species (not by image)."""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def stratified_species_split(
    manifest_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split manifest by species so all images of one species stay in the same fold.

    Uses GroupShuffleSplit with species as groups. First splits off test,
    then splits remainder into train/val.

    Returns:
        (train_df, val_df, test_df) each with a 'split' column.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    df = manifest_df.copy()
    groups = df["species"]

    # Split off test set
    splitter_test = GroupShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=random_state
    )
    trainval_idx, test_idx = next(splitter_test.split(df, groups=groups))
    test_df = df.iloc[test_idx].copy()
    trainval_df = df.iloc[trainval_idx].copy()

    # Split remainder into train/val
    relative_val = val_ratio / (train_ratio + val_ratio)
    splitter_val = GroupShuffleSplit(
        n_splits=1, test_size=relative_val, random_state=random_state
    )
    trainval_groups = trainval_df["species"]
    train_idx, val_idx = next(
        splitter_val.split(trainval_df, groups=trainval_groups)
    )
    train_df = trainval_df.iloc[train_idx].copy()
    val_df = trainval_df.iloc[val_idx].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Verify no species leakage
    train_sp = set(train_df["species"])
    val_sp = set(val_df["species"])
    test_sp = set(test_df["species"])
    assert not (train_sp & val_sp), "Species leakage: train/val overlap"
    assert not (train_sp & test_sp), "Species leakage: train/test overlap"
    assert not (val_sp & test_sp), "Species leakage: val/test overlap"

    print(
        f"Split: train={len(train_df)} ({len(train_sp)} spp), "
        f"val={len(val_df)} ({len(val_sp)} spp), "
        f"test={len(test_df)} ({len(test_sp)} spp)"
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

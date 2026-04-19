"""Per-head training loop on cached embeddings with early stopping."""

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from db.connection import get_session
from vision.data.dataset import MushroomFeatureDataset
from vision.data.manifest import build_manifest
from vision.data.splits import stratified_species_split
from vision.models.heads import FeatureHead
from vision.tracking.mlflow_utils import log_training_run, setup_experiment
from vision.training.losses import build_loss
from vision.training.metrics import (
    classification_report_str,
    compute_metrics,
)

VISION_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = VISION_ROOT / "config"


def _filter_rare_classes(manifest_df, min_species: int) -> tuple:
    """Remove classes with fewer than min_species unique species.

    Returns (filtered_df, dropped_classes) where dropped_classes is a list
    of (class_name, n_species, n_images) tuples.
    """
    species_per_class = manifest_df.groupby("feature_value")["species"].nunique()
    rare = species_per_class[species_per_class < min_species]

    if rare.empty:
        return manifest_df, []

    dropped = []
    for cls_name, n_spp in rare.items():
        n_imgs = len(manifest_df[manifest_df["feature_value"] == cls_name])
        dropped.append((cls_name, n_spp, n_imgs))

    filtered = manifest_df[~manifest_df["feature_value"].isin(rare.index)].reset_index(drop=True)
    return filtered, dropped


def _build_balanced_sampler(dataset: MushroomFeatureDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that balances classes within each epoch."""
    labels = torch.tensor(dataset._labels)
    class_counts = torch.bincount(labels, minlength=dataset.num_classes).float()
    # Sqrt-dampened balancing: partial rebalance without overcompensating
    sample_weights = 1.0 / class_counts[labels].sqrt()
    return WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)


@dataclass
class TrainingResult:
    feature_name: str
    best_epoch: int
    best_val_f1: float
    test_metrics: dict
    checkpoint_path: str


def _load_config() -> dict:
    with open(CONFIG_DIR / "training.yaml") as f:
        return yaml.safe_load(f)


def _load_feature_config(feature_name: str) -> dict:
    with open(CONFIG_DIR / "features.yaml") as f:
        features = yaml.safe_load(f)
    return features[feature_name]


def _train_one_epoch(head, loader, loss_fn, optimizer, device):
    head.train()
    total_loss = 0.0
    for embeddings, labels, _ in loader:
        embeddings = embeddings.to(device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        optimizer.zero_grad()
        logits = head(embeddings)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _evaluate(head, loader, device) -> tuple[list[int], list[int]]:
    head.eval()
    all_preds, all_labels = [], []
    for embeddings, labels, _ in loader:
        embeddings = embeddings.to(device)
        logits = head(embeddings)
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


def train_head(
    feature_name: str,
    preprocess_version: str = "v1",
    lr: float | None = None,
    weight_decay: float | None = None,
    dropout: float | None = None,
    label_smoothing: float | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    batch_size: int | None = None,
    min_species_per_class: int | None = None,
    device: str | None = None,
    quality_filter: bool = True,
) -> TrainingResult:
    """Train a single feature head end-to-end.

    1. Build manifest from DB
    2. Stratified species split
    3. Create datasets from cached embeddings
    4. Compute class weights, init head, optimizer, loss
    5. Training loop with early stopping on val macro-F1
    6. Save checkpoint, log to MLflow
    """
    config = _load_config()
    feat_cfg = _load_feature_config(feature_name)

    # Resolve params with defaults
    device = device or config["device"]
    lr = lr or config["default_lr"]
    weight_decay = weight_decay or config["default_weight_decay"]
    dropout = dropout if dropout is not None else config["default_dropout"]
    label_smoothing = (
        label_smoothing if label_smoothing is not None else config["default_label_smoothing"]
    )
    max_epochs = max_epochs or config["max_epochs"]
    patience = patience or config["early_stopping_patience"]
    batch_size = batch_size or config.get("default_batch_size", 64)
    min_species = (
        min_species_per_class
        if min_species_per_class is not None
        else config.get("min_species_per_class", 3)
    )

    class_names = feat_cfg["classes"]
    embedding_dir = (
        Path(config["paths"]["embeddings"])
        / "facebook-dinov2-base"
        / f"processed-{preprocess_version}"
    )

    # Build manifest and split
    with get_session() as session:
        manifest = build_manifest(
            session,
            feature_name,
            embedding_dir,
            preprocess_version,
            quality_filter=quality_filter,
        )

    # Filter rare classes (too few species for species-level split)
    if min_species > 1:
        manifest, dropped = _filter_rare_classes(manifest, min_species)
        if dropped:
            for cls_name, n_spp, n_imgs in dropped:
                print(f"  Dropped class '{cls_name}' ({n_spp} species, {n_imgs} images)")

    # Always sync class_names with what's actually in the manifest
    # (handles both rare-class filtering AND classes with 0 images)
    present = set(manifest["feature_value"])
    class_names = [c for c in class_names if c in present]

    train_df, val_df, test_df = stratified_species_split(manifest)

    train_ds = MushroomFeatureDataset(train_df, class_names)
    val_ds = MushroomFeatureDataset(val_df, class_names)
    test_ds = MushroomFeatureDataset(test_df, class_names)

    # Balanced sampling — each batch sees roughly equal class representation
    train_sampler = _build_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Head, optimizer, loss (balanced sampler handles class imbalance — no class weights in loss)
    head = FeatureHead(
        embed_dim=config["embedding_dim"],
        num_classes=len(class_names),
        feature_name=feature_name,
        class_names=class_names,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    loss_fn = build_loss(None, label_smoothing, device)

    # Training loop
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None

    print(
        f"\nTraining {feature_name} head ({len(train_ds)}/{len(val_ds)}/{len(test_ds)} "
        f"train/val/test, {len(class_names)} classes)"
    )
    print(
        f"  lr={lr}, wd={weight_decay}, dropout={dropout}, "
        f"label_smoothing={label_smoothing}, device={device}"
    )

    for epoch in range(1, max_epochs + 1):
        train_loss = _train_one_epoch(head, train_loader, loss_fn, optimizer, device)
        val_preds, val_labels = _evaluate(head, val_loader, device)
        val_metrics = compute_metrics(val_preds, val_labels, class_names)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch <= 5 or epoch % 10 == 0 or epochs_without_improvement == 0:
            print(
                f"  Epoch {epoch:3d}: loss={train_loss:.4f}  "
                f"val_acc={val_metrics['accuracy']:.3f}  "
                f"val_f1={val_metrics['macro_f1']:.3f}"
                f"{'  *best*' if epochs_without_improvement == 0 else ''}"
            )

        scheduler.step()

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best model and evaluate on test
    head.load_state_dict(best_state)
    head.to(device)
    test_preds, test_labels = _evaluate(head, test_loader, device)
    test_metrics = compute_metrics(test_preds, test_labels, class_names)

    print(f"\nTest results (best epoch {best_epoch}):")
    print(
        f"  accuracy={test_metrics['accuracy']:.3f}  "
        f"macro_f1={test_metrics['macro_f1']:.3f}  "
        f"weighted_f1={test_metrics['weighted_f1']:.3f}"
    )
    print(classification_report_str(test_preds, test_labels, class_names))

    # Save checkpoint
    ckpt_dir = Path(config["paths"]["checkpoints"]) / feature_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / "best.pt")
    head.save(ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    # Log to MLflow
    experiment_id = setup_experiment(feature_name)
    log_training_run(
        experiment_id=experiment_id,
        params={
            "feature_name": feature_name,
            "num_classes": len(class_names),
            "tier": feat_cfg["tier"],
            "preprocess_version": preprocess_version,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "label_smoothing": label_smoothing,
            "batch_size": batch_size,
            "min_species_per_class": min_species,
            "best_epoch": best_epoch,
            "device": device,
        },
        metrics={
            "best_val_f1": best_val_f1,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
        },
    )

    return TrainingResult(
        feature_name=feature_name,
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        test_metrics=test_metrics,
        checkpoint_path=ckpt_path,
    )

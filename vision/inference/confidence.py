"""Temperature scaling calibration and confidence thresholding per head."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vision.data.dataset import MushroomFeatureDataset
from vision.models.heads import FeatureHead


def calibrate_head(
    head: FeatureHead,
    val_dataset: MushroomFeatureDataset,
    lr: float = 0.01,
    max_iter: int = 100,
    device: str = "cpu",
) -> float:
    """Learn optimal temperature via NLL minimization on validation set.

    Temperature scaling: softmax(logits / T) where T is learned.
    Higher T -> softer (less confident) predictions.
    Lower T -> sharper (more confident) predictions.

    Args:
        head: Trained FeatureHead (frozen, eval mode).
        val_dataset: Validation dataset with cached embeddings.
        lr: Learning rate for temperature optimization.
        max_iter: Max optimization steps.
        device: Device for computation.

    Returns:
        Optimal temperature value.
    """
    head.eval()
    head.to(device)

    # Collect all logits and labels from val set
    loader = DataLoader(val_dataset, batch_size=256)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for embeddings, labels, _ in loader:
            logits = head(embeddings.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(torch.as_tensor(labels, dtype=torch.long))

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    # Optimize temperature
    log_temp = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temp = log_temp.exp()
        loss = F.cross_entropy(logits / temp, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = log_temp.exp().item()

    # Update head
    head.temperature = temperature
    return temperature


def reliability_stats(
    head: FeatureHead,
    dataset: MushroomFeatureDataset,
    n_bins: int = 10,
    device: str = "cpu",
) -> dict:
    """Compute reliability diagram data: per-bin confidence vs accuracy.

    Returns:
        dict with keys: bin_edges, bin_confidences, bin_accuracies, bin_counts, ece.
    """
    head.eval()
    head.to(device)
    loader = DataLoader(dataset, batch_size=256)

    all_confs, all_correct = [], []
    with torch.no_grad():
        for embeddings, labels, _ in loader:
            logits = head(embeddings.to(device))
            probs = F.softmax(logits / head.temperature, dim=-1)
            top_conf, top_pred = probs.max(dim=-1)
            labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
            correct = (top_pred == labels_t).float()
            all_confs.append(top_conf.cpu())
            all_correct.append(correct.cpu())

    confs = torch.cat(all_confs)
    correct = torch.cat(all_correct)

    bin_edges = torch.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        count = mask.sum().item()
        bin_counts.append(count)
        if count > 0:
            avg_conf = confs[mask].mean().item()
            avg_acc = correct[mask].mean().item()
            bin_confidences.append(avg_conf)
            bin_accuracies.append(avg_acc)
            ece += count * abs(avg_acc - avg_conf)
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)

    ece /= len(confs)

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "ece": ece,
    }


def find_confidence_threshold(
    head: FeatureHead,
    dataset: MushroomFeatureDataset,
    target_accuracy: float = 0.85,
    device: str = "cpu",
) -> float | None:
    """Find the minimum confidence threshold that achieves target accuracy.

    Returns:
        Threshold value, or None if target accuracy is not achievable.
    """
    head.eval()
    head.to(device)
    loader = DataLoader(dataset, batch_size=256)

    all_confs, all_correct = [], []
    with torch.no_grad():
        for embeddings, labels, _ in loader:
            logits = head(embeddings.to(device))
            probs = F.softmax(logits / head.temperature, dim=-1)
            top_conf, top_pred = probs.max(dim=-1)
            labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
            correct = (top_pred == labels_t).float()
            all_confs.append(top_conf.cpu())
            all_correct.append(correct.cpu())

    confs = torch.cat(all_confs)
    correct = torch.cat(all_correct)

    # Try thresholds from high to low
    for threshold in torch.arange(0.95, 0.1, -0.05):
        mask = confs >= threshold
        if mask.sum() < 10:
            continue
        acc = correct[mask].mean().item()
        if acc >= target_accuracy:
            return threshold.item()

    return None

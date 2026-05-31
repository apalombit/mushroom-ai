"""Per-class accuracy, macro-F1, confusion matrix, classification report."""

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    preds: list[int], labels: list[int], class_names: list[str]
) -> dict:
    """Compute accuracy, macro-F1, and weighted-F1."""
    preds_arr = np.array(preds)
    labels_arr = np.array(labels)
    accuracy = (preds_arr == labels_arr).mean()
    macro_f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def classification_report_str(
    preds: list[int], labels: list[int], class_names: list[str]
) -> str:
    """Sklearn classification report as a formatted string."""
    present_labels = sorted(set(labels) | set(preds))
    present_names = [class_names[i] for i in present_labels]
    return classification_report(
        labels, preds, labels=present_labels, target_names=present_names, zero_division=0
    )


def compute_confusion_matrix(
    preds: list[int], labels: list[int], class_names: list[str]
) -> np.ndarray:
    """Return confusion matrix array (true × predicted)."""
    return confusion_matrix(labels, preds, labels=list(range(len(class_names))))

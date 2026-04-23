import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader


def get_predictions(
    model,
    dataset,
    threshold:  float = 0.5,
    batch_size: int   = 4,
) -> tuple[list, list]:
    """
    Run model over a dataset and return (predictions, ground_truth).
    Keeps everything on CPU after each batch to avoid OOM.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader     = DataLoader(dataset, batch_size=batch_size)
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in loader:
            logits = model(list(texts))
            probs  = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_labels.append(labels.cpu())
            torch.cuda.empty_cache()

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds  = (probs >= threshold).astype(int)

    return preds.tolist(), labels.tolist(), probs.tolist()


def plot_confusion_matrix(
    preds:      list,
    labels:     list,
    model_name: str,
    save_path:  str = None,
) -> dict:
    """
    Plot and optionally save a confusion matrix.
    Returns a dict of metrics.
    """
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))

    matrix = np.array([[tn, fp],
                        [fn, tp]])

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (tp + tn) / (tp + tn + fp + fn)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted: other", "Predicted: trump"],
        yticklabels=["Actual: other",    "Actual: trump"],
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_title(f"{model_name} — test set", fontsize=13, pad=12)
    ax.set_xlabel(
        f"Accuracy: {accuracy:.3f}   Precision: {precision:.3f}   "
        f"Recall: {recall:.3f}   F1: {f1:.3f}",
        fontsize=9, labelpad=10
    )

    plt.tight_layout()
    save_path = f'./{model_name}.png'
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
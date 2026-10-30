"""
common/evaluation.py
--------------------
Performance evaluation shared by all three solutions, so every method is
scored and plotted in exactly the same way.

Metrics
    accuracy, macro precision, macro recall, macro F1, macro ROC-AUC
    ("macro" = computed per class then averaged, so small classes count as
    much as big ones), plus the miss rate: diseased leaves called healthy.

Plots
    confusion matrix, one-vs-rest ROC curves, per-class F1 bars.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from common.dataset import find_healthy_class, pretty_class_name

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def compute_metrics(y_true, y_pred, scores=None, class_names=None):
    """Return a dict of summary metrics.

    Parameters
    ----------
    scores : array (n_samples, n_classes), optional
        Class probabilities / scores, needed for ROC-AUC.
    """
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if scores is not None:
        n_classes = scores.shape[1]
        present = np.unique(y_true)
        try:
            if n_classes == 2:
                m["roc_auc_macro"] = float(roc_auc_score(y_true, scores[:, 1]))
            elif len(present) == n_classes:
                m["roc_auc_macro"] = float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro"))
        except ValueError:
            pass

    # Miss rate: fraction of diseased leaves predicted as healthy (the costly error)
    if class_names is not None:
        h = find_healthy_class(class_names)
        if h is not None:
            diseased = y_true != h
            if diseased.any():
                m["miss_rate"] = float(np.mean(y_pred[diseased] == h))
    return m


def report_text(y_true, y_pred, class_names):
    """Per-class precision / recall / F1 table as plain text."""
    labels = list(range(len(class_names)))
    return classification_report(y_true, y_pred, labels=labels,
                                 target_names=[pretty_class_name(c) for c in class_names],
                                 zero_division=0)


def plot_confusion_matrix(y_true, y_pred, class_names, path, title="Confusion matrix"):
    """Row-normalised confusion matrix (each row sums to 1 = recall per class)."""
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = [pretty_class_name(c) for c in class_names]

    size = max(4.5, 0.55 * len(names) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels, names, rotation=45, ha="right")
    ax.set_yticks(labels, names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for i in labels:
        for j in labels:
            if cm[i, j]:
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc_curves(y_true, scores, class_names, path, title="ROC curves (one-vs-rest)"):
    """One ROC curve per class (that class = positive, all others = negative)."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2:
        y_bin = np.column_stack([1 - y_bin[:, 0], y_bin[:, 0]])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
    for c in range(n_classes):
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], scores[:, c])
        ax.plot(fpr, tpr, lw=1.4, color=cmap(c % cmap.N),
                label=f"{pretty_class_name(class_names[c])} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_per_class_f1(y_true, y_pred, class_names, path, title="Per-class F1 score"):
    labels = list(range(len(class_names)))
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    names = [pretty_class_name(c) for c in class_names]
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(names) + 2), 3.6))
    ax.bar(names, f1, color="#4C72B0")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    for i, v in enumerate(f1):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def full_evaluation(name, y_true, y_pred, scores, class_names, out_dir):
    """Compute metrics, write the text report and draw the three standard plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred, scores, class_names)
    (out_dir / f"{name}_report.txt").write_text(report_text(y_true, y_pred, class_names))
    plot_confusion_matrix(y_true, y_pred, class_names, out_dir / f"{name}_confusion_matrix.png",
                          title=f"Confusion matrix - {name}")
    if scores is not None:
        plot_roc_curves(y_true, scores, class_names, out_dir / f"{name}_roc.png",
                        title=f"ROC curves - {name}")
    plot_per_class_f1(y_true, y_pred, class_names, out_dir / f"{name}_per_class_f1.png",
                      title=f"Per-class F1 - {name}")
    return metrics


def save_results(results, path):
    """Write the results dictionary as pretty JSON (read later by compare_solutions.py)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def print_metrics(name, m):
    parts = [f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)]
    print(f"[{name}] " + "  ".join(parts))

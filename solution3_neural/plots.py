"""
solution3_neural/plots.py
-------------------------
Figures specific to Solution 3:

* training_curves        - loss and accuracy per epoch (train vs. validation)
* hidden_representation  - the MLP's last hidden layer projected to 2-D,
                           showing how training pulls the classes apart
* model_comparison       - perceptron vs. MLP vs. CNN on the test set
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from common.dataset import pretty_class_name


def training_curves(history, path, title="Training curves", best_epoch=None):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="validation")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        if best_epoch:
            ax.axvline(best_epoch, color="grey", ls="--", lw=0.8)
        ax.legend(fontsize=7)
    if best_epoch:
        axes[0].text(best_epoch, axes[0].get_ylim()[1], " best", fontsize=7, va="top", color="grey")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def hidden_representation(mlp, X, y, class_names, path):
    """PCA of the input features vs. PCA of the last hidden layer's activations."""
    hidden = mlp._forward(X)[-2]               # activations just before the softmax
    cmap = plt.get_cmap("tab10" if len(class_names) <= 10 else "tab20")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, Z, title in ((axes[0], X, "Input features (PCA)"),
                         (axes[1], hidden, "Last hidden layer of the MLP (PCA)")):
        P = PCA(2).fit_transform(Z)
        for c in range(len(class_names)):
            ax.scatter(*P[y == c].T, s=8, alpha=0.65, color=cmap(c % cmap.N),
                       label=pretty_class_name(class_names[c]))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(fontsize=6.5, markerscale=1.5)
    fig.suptitle("What the network learned: classes become separable in the hidden layer", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def model_comparison(results, path):
    """Grouped bars of accuracy and macro F1 for each neural model."""
    names = list(results)
    acc = [results[n]["accuracy"] for n in names]
    f1 = [results[n]["f1_macro"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(x - 0.18, acc, 0.36, label="Accuracy", color="#4C72B0")
    ax.bar(x + 0.18, f1, 0.36, label="Macro F1", color="#55A868")
    for i in range(len(names)):
        ax.text(x[i] - 0.18, acc[i] + 0.01, f"{acc[i]:.3f}", ha="center", fontsize=7)
        ax.text(x[i] + 0.18, f1[i] + 0.01, f"{f1[i]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.08)
    ax.set_title("Neural models on the test set")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

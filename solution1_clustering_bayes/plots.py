"""
solution1_clustering_bayes/plots.py
-----------------------------------
Figures specific to Solution 1:

* segmentation_examples  - original | leaf mask | k-means clusters | lesions
* feature_distributions  - how the key lesion features differ per class
* risk_tradeoff          - accuracy vs. miss rate as the miss cost grows
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.dataset import pretty_class_name
from common.features import LESION_FEATURE_NAMES, lesion_features


def segmentation_examples(images, labels, class_names, path, per_class=1, seed=0):
    """Grid showing every stage of the unsupervised segmentation, one row per example."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(len(class_names)):
        idx = np.flatnonzero(labels == c)
        if len(idx):
            rows.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False))

    fig, axes = plt.subplots(len(rows), 4, figsize=(7.2, 1.85 * len(rows)))
    axes = np.atleast_2d(axes)
    titles = ["Original", "Leaf mask (Otsu)", "K-means colour clusters", "Detected lesions"]
    for r, i in enumerate(rows):
        img = images[i]
        _, leaf, lesion, cmap = lesion_features(img, return_masks=True)
        overlay = img.copy()
        overlay[lesion] = (0.45 * overlay[lesion] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
        cl = np.ma.masked_where(cmap < 0, cmap)
        panels = [img, leaf, cl, overlay]
        for c, ax in enumerate(axes[r]):
            if c == 1:
                ax.imshow(panels[c], cmap="gray")
            elif c == 2:
                ax.imshow(np.zeros_like(img))
                ax.imshow(panels[c], cmap="viridis")
            else:
                ax.imshow(panels[c])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
        axes[r, 0].set_ylabel(pretty_class_name(class_names[labels[i]]), fontsize=7)
        frac = lesion.sum() / max(leaf.sum(), 1)
        axes[r, 3].set_xlabel(f"lesion area {frac:.0%}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def feature_distributions(X, y, class_names, path,
                          features=("lesion_area_ratio", "n_spots", "brown_ratio", "yellow_ratio")):
    """Box plots of a few interpretable lesion features, split by class."""
    names = [pretty_class_name(c) for c in class_names]
    fig, axes = plt.subplots(1, len(features), figsize=(3.0 * len(features), 3.4))
    for ax, f in zip(np.atleast_1d(axes), features):
        j = LESION_FEATURE_NAMES.index(f)
        data = [X[y == c, j] for c in range(len(class_names))]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(names) + 1), names, rotation=60, ha="right", fontsize=7)
        ax.set_title(f.replace("_", " "), fontsize=9)
    fig.suptitle("Lesion features per class (from k-means segmentation)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def risk_tradeoff(costs, accuracies, miss_rates, path):
    """How raising the cost of a missed disease trades overall accuracy for fewer misses."""
    fig, ax1 = plt.subplots(figsize=(5.2, 3.6))
    ax1.plot(costs, accuracies, "o-", color="#4C72B0", label="Accuracy")
    ax1.set_xlabel("Cost of calling a diseased leaf 'healthy'")
    ax1.set_ylabel("Accuracy", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(costs, miss_rates, "s--", color="#C44E52", label="Miss rate")
    ax2.set_ylabel("Miss rate (diseased -> healthy)", color="#C44E52")
    ax2.spines["right"].set_visible(True)
    ax1.set_xscale("log")
    ax1.set_title("Minimum-risk Bayes decision: accuracy vs. missed diseases")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

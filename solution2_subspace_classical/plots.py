"""
solution2_subspace_classical/plots.py
-------------------------------------
Figures specific to Solution 2:

* feature_ranking        - top features by Fisher score and information gain
* pca_variance           - scree plot + cumulative explained variance
* projection_scatter     - training leaves projected to 2-D by PCA and by LDA
* curve                  - test accuracy vs. a hyper-parameter
* cv_comparison          - cross-validated accuracy of every candidate pipeline
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.dataset import pretty_class_name


def feature_ranking(fisher, info_gain, names, path, top=20):
    """Side-by-side bar charts of the best features under the two criteria."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, scores, title, col in ((axes[0], fisher, "Fisher score", "#4C72B0"),
                                   (axes[1], info_gain, "Information gain (mutual information)", "#55A868")):
        order = np.argsort(scores)[::-1][:top][::-1]
        ax.barh([names[i] for i in order], scores[order], color=col)
        ax.set_title(f"Top {top} features - {title}", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pca_variance(pca, path):
    """Individual and cumulative explained variance of the PCA components."""
    ratio = pca.explained_variance_ratio_
    cum = np.cumsum(ratio)
    n = np.arange(1, len(ratio) + 1)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(n, ratio, color="#8DA0CB", label="per component")
    ax.plot(n, cum, "o-", color="#C44E52", ms=3, label="cumulative")
    ax.axhline(0.95, color="grey", ls="--", lw=0.8)
    ax.text(n[-1], 0.955, "95 %", ha="right", va="bottom", fontsize=7, color="grey")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA explained variance")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def projection_scatter(Z_pca, Z_lda, y, class_names, path):
    """2-D scatter of the PCA and LDA projections, one colour per class.

    LDA usually shows much tighter, better separated class clusters because
    it uses the labels, while PCA only follows overall variance.
    """
    cmap = plt.get_cmap("tab10" if len(class_names) <= 10 else "tab20")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, Z, title in ((axes[0], Z_pca, "PCA (unsupervised)"), (axes[1], Z_lda, "LDA (supervised)")):
        for c in range(len(class_names)):
            pts = Z[y == c]
            if len(pts) and Z.shape[1] >= 2:
                ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.65, color=cmap(c % cmap.N),
                           label=pretty_class_name(class_names[c]))
            elif len(pts):     # 2-class LDA gives a single axis
                ax.scatter(pts[:, 0], np.random.default_rng(c).normal(0, 0.05, len(pts)) + c,
                           s=8, alpha=0.65, color=cmap(c % cmap.N), label=pretty_class_name(class_names[c]))
        ax.set_title(title)
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
    axes[1].legend(fontsize=6.5, loc="best", markerscale=1.5)
    fig.suptitle("Leaf feature vectors projected to two dimensions", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def curve(xs, ys, xlabel, title, path, ylabel="Test accuracy"):
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.plot(xs, ys, "o-", color="#4C72B0")
    best = int(np.argmax(ys))
    ax.annotate(f"best {ys[best]:.3f} at {xs[best]}", (xs[best], ys[best]),
                textcoords="offset points", xytext=(0, -16), ha="center", fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cv_comparison(cv_results, path):
    """Mean +/- std 5-fold CV accuracy of each pipeline (the chosen one highlighted)."""
    names = list(cv_results)
    means = [cv_results[n]["mean"] for n in names]
    stds = [cv_results[n]["std"] for n in names]
    best = int(np.argmax(means))
    colors = ["#C44E52" if i == best else "#8DA0CB" for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(names, means, yerr=stds, capsize=3, color=colors)
    lo = max(0.0, min(means) - 2 * max(stds) - 0.05)
    ax.set_ylim(lo, 1.0)
    ax.set_ylabel("5-fold CV accuracy (train set)")
    ax.set_title("Model selection for Solution 2 (red = selected)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

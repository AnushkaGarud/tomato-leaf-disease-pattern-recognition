"""
solution2_subspace_classical/feature_selection.py
-------------------------------------------------
Feature evaluation criteria (Unit III) and a FILTER-method selector.

Fisher score of feature j
    F_j = sum_c n_c (mu_cj - mu_j)^2  /  sum_c n_c sigma_cj^2

    numerator   = between-class scatter (how far the class means are apart)
    denominator = within-class scatter  (how spread each class is)
    A high score means the feature alone separates the classes well.

Information gain (mutual information) I(X_j ; Y)
    How much knowing feature j reduces uncertainty about the class.
    Unlike Fisher score it also catches non-linear relationships.

FisherScoreSelector is a scikit-learn transformer, so it can sit inside a
Pipeline and is re-fitted on each cross-validation fold (no data leakage).
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

import config


def fisher_score(X, y):
    """Fisher score of every column of X -> array (n_features,)."""
    mu = X.mean(axis=0)
    between = np.zeros(X.shape[1])
    within = np.zeros(X.shape[1])
    for c in np.unique(y):
        Xc = X[y == c]
        between += len(Xc) * (Xc.mean(axis=0) - mu) ** 2
        within += len(Xc) * Xc.var(axis=0)
    return between / np.maximum(within, 1e-12)


def information_gain(X, y, seed=config.RANDOM_STATE):
    """Mutual information between each feature and the class label."""
    return mutual_info_classif(X, y, random_state=seed)


class FisherScoreSelector(BaseEstimator, TransformerMixin):
    """Keep the k features with the highest Fisher score (filter method)."""

    def __init__(self, k=config.TOP_K_FEATURES):
        self.k = k

    def fit(self, X, y):
        self.scores_ = fisher_score(X, y)
        k = min(self.k, X.shape[1])
        self.selected_ = np.sort(np.argsort(self.scores_)[::-1][:k])
        return self

    def transform(self, X):
        return X[:, self.selected_]

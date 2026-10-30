"""
solution1_clustering_bayes/bayes_classifier.py
----------------------------------------------
A Gaussian Bayes classifier written from scratch with NumPy.

Theory (Unit II)
    Bayes theorem:   P(w_i | x) = p(x | w_i) P(w_i) / p(x)

    * Class-conditional density p(x | w_i) is modelled as a multivariate
      Gaussian N(mu_i, Sigma_i) estimated from the training samples of class i.
    * Prior P(w_i) = fraction of training samples in class i.
    * p(x) is the same for every class, so it only normalises the posterior.

    Minimum-error-rate rule (MAP):  choose argmax_i P(w_i | x)
    Minimum-risk rule:              choose argmin_j  sum_i L(j, i) P(w_i | x)
        where L(j, i) is the loss of deciding w_j when the truth is w_i.

    With a separate Sigma_i per class the decision boundaries are QUADRATIC
    (a quadratic discriminant). With naive=True every Sigma_i is diagonal
    (features assumed independent given the class) = Gaussian Naive Bayes.

Everything is done in log space for numerical stability.
"""

import numpy as np


class GaussianBayesClassifier:
    """Bayes classifier with Gaussian class-conditional densities.

    Parameters
    ----------
    naive : bool
        True  -> diagonal covariances (Naive Bayes, linear-ish boundaries)
        False -> full covariances (quadratic discriminant)
    reg : float
        Added to the covariance diagonal so it is always invertible
        (important when features are correlated or a class has few samples).
    """

    def __init__(self, naive=False, reg=1e-3):
        self.naive = naive
        self.reg = reg

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n, d = X.shape
        self.means_, self.log_priors_ = [], []
        self._inv_covs, self._log_dets = [], []

        for c in self.classes_:
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            if self.naive:
                var = Xc.var(axis=0) + self.reg
                cov = np.diag(var)
            else:
                cov = np.cov(Xc, rowvar=False) if len(Xc) > 1 else np.eye(d)
                cov = np.atleast_2d(cov) + self.reg * np.eye(d)
            sign, logdet = np.linalg.slogdet(cov)
            self.means_.append(mu)
            self._inv_covs.append(np.linalg.inv(cov))
            self._log_dets.append(logdet)
            self.log_priors_.append(np.log(len(Xc) / n))

        self.means_ = np.array(self.means_)
        self.log_priors_ = np.array(self.log_priors_)
        return self

    # --------------------------------------------------------- discriminants
    def log_likelihood(self, X):
        """log p(x | w_i) for every sample and class -> (n_samples, n_classes)."""
        d = X.shape[1]
        out = np.empty((len(X), len(self.classes_)))
        for k in range(len(self.classes_)):
            diff = X - self.means_[k]
            # Mahalanobis distance (x - mu)^T Sigma^-1 (x - mu)
            maha = np.einsum("ij,jk,ik->i", diff, self._inv_covs[k], diff)
            out[:, k] = -0.5 * (maha + self._log_dets[k] + d * np.log(2 * np.pi))
        return out

    def discriminant(self, X):
        """g_i(x) = log p(x | w_i) + log P(w_i)   (unnormalised log posterior)."""
        return self.log_likelihood(X) + self.log_priors_

    def predict_proba(self, X):
        """Posterior P(w_i | x) via Bayes theorem (softmax of the discriminants)."""
        g = self.discriminant(X)
        g -= g.max(axis=1, keepdims=True)          # log-sum-exp trick
        p = np.exp(g)
        return p / p.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------ decisions
    def predict(self, X):
        """Minimum-error-rate decision: the class with the largest posterior."""
        return self.classes_[np.argmax(self.discriminant(X), axis=1)]

    def predict_min_risk(self, X, loss):
        """Minimum-risk decision.

        loss[j, i] = cost of deciding class j when the true class is i.
        Conditional risk R(j | x) = sum_i loss[j, i] * P(w_i | x);
        we pick the decision j with the smallest risk.
        """
        risk = self.predict_proba(X) @ np.asarray(loss).T     # (n, n_decisions)
        return self.classes_[np.argmin(risk, axis=1)]


def miss_cost_loss_matrix(n_classes, healthy_idx, miss_cost):
    """Zero-one loss, except that calling a diseased leaf 'healthy' costs `miss_cost`.

    Missing a disease lets it spread through the field, while a false alarm
    only costs a closer look, so the two errors should not be weighted equally.
    """
    loss = 1.0 - np.eye(n_classes)
    if healthy_idx is not None:
        for i in range(n_classes):
            if i != healthy_idx:
                loss[healthy_idx, i] = miss_cost       # decide healthy, truth diseased
    return loss

"""
solution3_neural/mlp_numpy.py
-----------------------------
A Perceptron and a Multilayer Perceptron written from scratch in NumPy, so
every step of the learning algorithm is visible (Unit V).

Perceptron (Rosenblatt, multi-class version)
    One weight vector per class, score_c = w_c . x + b_c, predict argmax.
    On a mistake: move the true class's weights TOWARDS x and the wrongly
    predicted class's weights AWAY from x. It can only learn linear
    boundaries, so it serves as the baseline.

Multilayer Perceptron + Backpropagation
    Forward pass   a_0 = x
                   z_l = a_{l-1} W_l + b_l
                   a_l = ReLU(z_l)               (hidden layers)
                   p   = softmax(z_L)            (output layer)
    Loss           cross-entropy  -sum y log p   + L2 weight penalty
    Backward pass  delta_L = p - y               (softmax + cross-entropy)
                   delta_l = (delta_{l+1} W_{l+1}^T) * ReLU'(z_l)
                   dW_l = a_{l-1}^T delta_l,  db_l = sum delta_l
    Update         mini-batch SGD with momentum:
                   v = mu v - lr dW ;  W = W + v
    Early stopping keeps the weights with the lowest validation loss.
"""

import numpy as np
from sklearn.model_selection import train_test_split

import config


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _one_hot(idx, k):
    out = np.zeros((len(idx), k))
    out[np.arange(len(idx)), idx] = 1.0
    return out


# ==========================================================================
# Perceptron
# ==========================================================================
class Perceptron:
    """Multi-class (averaged) perceptron trained with the perceptron rule."""

    def __init__(self, epochs=50, lr=1.0, seed=config.RANDOM_STATE):
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        Xb = np.hstack([X, np.ones((len(X), 1))])          # bias as an extra input
        W = np.zeros((len(self.classes_), Xb.shape[1]))
        W_sum = np.zeros_like(W)                            # for the averaged perceptron
        self.history_ = {"train_acc": []}
        for _ in range(self.epochs):
            mistakes = 0
            for i in rng.permutation(len(Xb)):
                pred = np.argmax(W @ Xb[i])
                if pred != yi[i]:                           # perceptron update rule
                    W[yi[i]] += self.lr * Xb[i]
                    W[pred] -= self.lr * Xb[i]
                    mistakes += 1
                W_sum += W
            self.history_["train_acc"].append(1 - mistakes / len(Xb))
        # Averaging the weights over all steps is far more stable than the last W
        self.W_ = W_sum / (self.epochs * len(Xb))
        return self

    def decision_function(self, X):
        return np.hstack([X, np.ones((len(X), 1))]) @ self.W_.T

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X):
        """Not a true probability - softmax of scores, used only for ROC curves."""
        s = self.decision_function(X)
        return _softmax(s / (np.abs(s).std() + 1e-9))


# ==========================================================================
# Multilayer Perceptron
# ==========================================================================
class MLP:
    """Fully connected network, ReLU hidden layers, softmax output.

    Parameters
    ----------
    hidden : tuple of int   sizes of the hidden layers
    lr, momentum, l2        SGD settings
    epochs, batch_size      training length
    patience                early stopping: stop after this many epochs
                            without a new best validation loss
    """

    def __init__(self, hidden=config.MLP_HIDDEN, lr=config.MLP_LR, epochs=config.MLP_EPOCHS,
                 batch_size=config.MLP_BATCH, l2=config.MLP_L2, momentum=config.MLP_MOMENTUM,
                 patience=config.MLP_PATIENCE, seed=config.RANDOM_STATE, verbose=True):
        self.hidden, self.lr, self.epochs = tuple(hidden), lr, epochs
        self.batch_size, self.l2, self.momentum = batch_size, l2, momentum
        self.patience, self.seed, self.verbose = patience, seed, verbose

    # ------------------------------------------------------------ helpers
    def _init_weights(self, n_in, n_out, rng):
        sizes = [n_in, *self.hidden, n_out]
        # He initialisation: variance 2 / fan_in, suited to ReLU
        self.W = [rng.normal(0, np.sqrt(2.0 / a), size=(a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def _forward(self, X):
        """Return the list of activations a_0 .. a_L (last = softmax output)."""
        acts = [X]
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(_softmax(z) if l == len(self.W) - 1 else np.maximum(z, 0.0))
        return acts

    def _loss(self, P, Y):
        ce = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))
        return ce + 0.5 * self.l2 * sum(np.sum(W * W) for W in self.W)

    def _backward(self, acts, Y):
        """Backpropagation: gradients of the loss for every W and b."""
        n = len(Y)
        delta = (acts[-1] - Y) / n                   # dLoss/dz at the softmax layer
        grads_W, grads_b = [], []
        for l in range(len(self.W) - 1, -1, -1):
            grads_W.insert(0, acts[l].T @ delta + self.l2 * self.W[l])
            grads_b.insert(0, delta.sum(axis=0))
            if l > 0:                                # propagate to the previous layer
                delta = (delta @ self.W[l].T) * (acts[l] > 0)
        return grads_W, grads_b

    # --------------------------------------------------------------- fit
    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        if X_val is None:                             # hold out 10 % for early stopping
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, stratify=y,
                                                  random_state=self.seed)
        Y = _one_hot(np.searchsorted(self.classes_, y), k)
        Yv = _one_hot(np.searchsorted(self.classes_, y_val), k)

        self._init_weights(X.shape[1], k, rng)
        vel_W = [np.zeros_like(W) for W in self.W]
        vel_b = [np.zeros_like(b) for b in self.b]
        self.history_ = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best = (np.inf, None, 0)                      # (val loss, weights, epoch)

        for epoch in range(self.epochs):
            lr = self.lr * (0.5 ** (epoch // 50))     # halve the learning rate every 50 epochs
            order = rng.permutation(len(X))           # reshuffle every epoch
            for start in range(0, len(X), self.batch_size):
                idx = order[start:start + self.batch_size]
                acts = self._forward(X[idx])
                gW, gb = self._backward(acts, Y[idx])
                for l in range(len(self.W)):          # SGD with momentum
                    vel_W[l] = self.momentum * vel_W[l] - lr * gW[l]
                    vel_b[l] = self.momentum * vel_b[l] - lr * gb[l]
                    self.W[l] += vel_W[l]
                    self.b[l] += vel_b[l]

            # ---- bookkeeping for the learning curves
            P, Pv = self._forward(X)[-1], self._forward(X_val)[-1]
            h = self.history_
            h["train_loss"].append(self._loss(P, Y))
            h["val_loss"].append(self._loss(Pv, Yv))
            h["train_acc"].append(np.mean(P.argmax(1) == Y.argmax(1)))
            h["val_acc"].append(np.mean(Pv.argmax(1) == Yv.argmax(1)))

            if h["val_loss"][-1] < best[0] - 1e-5:
                best = (h["val_loss"][-1], ([W.copy() for W in self.W], [b.copy() for b in self.b]), epoch)
            elif epoch - best[2] >= self.patience:
                if self.verbose:
                    print(f"  [mlp] early stop at epoch {epoch + 1} (best epoch {best[2] + 1})")
                break
            if self.verbose and (epoch + 1) % 25 == 0:
                print(f"  [mlp] epoch {epoch + 1:3d}  loss {h['train_loss'][-1]:.4f}  "
                      f"val_loss {h['val_loss'][-1]:.4f}  val_acc {h['val_acc'][-1]:.4f}")

        self.W, self.b = best[1]
        self.best_epoch_ = best[2] + 1
        return self

    # ----------------------------------------------------------- predict
    def predict_proba(self, X):
        return self._forward(X)[-1]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ==========================================================================
# Gradient check (proves the backpropagation code is correct)
# ==========================================================================
def gradient_check(n=20, d=6, k=3, hidden=(5, 4), eps=1e-5, seed=0):
    """Compare backprop gradients with numerical finite differences.

    Returns the maximum relative error; values below ~1e-6 mean the
    analytic gradients are correct. Run:  python -m solution3_neural.mlp_numpy
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    Y = _one_hot(rng.integers(0, k, n), k)
    net = MLP(hidden=hidden, l2=1e-3, verbose=False)
    net._init_weights(d, k, rng)
    gW, _ = net._backward(net._forward(X), Y)
    worst = 0.0
    for l, W in enumerate(net.W):
        for idx in np.ndindex(W.shape):
            old = W[idx]
            W[idx] = old + eps
            plus = net._loss(net._forward(X)[-1], Y)
            W[idx] = old - eps
            minus = net._loss(net._forward(X)[-1], Y)
            W[idx] = old
            num = (plus - minus) / (2 * eps)
            rel = abs(num - gW[l][idx]) / max(1e-8, abs(num) + abs(gW[l][idx]))
            worst = max(worst, rel)
    return worst


if __name__ == "__main__":
    print(f"max relative gradient error: {gradient_check():.2e}  (should be < 1e-6)")

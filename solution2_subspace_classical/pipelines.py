"""
solution2_subspace_classical/pipelines.py
-----------------------------------------
The candidate models of Solution 2, each a scikit-learn Pipeline:

    standardise -> Fisher-score filter -> subspace projection -> classifier

Subspace projections (Unit III, feature extraction)
    PCA  - unsupervised: rotates the data onto the directions of maximum
           variance and keeps enough of them to explain 95 % of it.
    LDA  - supervised: finds at most (C - 1) directions that maximise
           between-class scatter relative to within-class scatter
           (Fisher's criterion). Ideal for classification.
    PCA -> LDA - the classic "Fisherfaces" recipe: PCA first removes noise
           and makes the within-class scatter matrix invertible, then LDA.

Classifiers
    kNN  - non-parametric, decides by majority vote of the k closest
           training leaves in the projected space.
    SVM  - RBF-kernel support vector machine, a max-margin classifier.

The best pipeline is chosen by 5-fold cross-validation on the TRAINING set
only; the test set is used once, at the very end.
"""

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from solution2_subspace_classical.feature_selection import FisherScoreSelector


def _knn():
    return KNeighborsClassifier(n_neighbors=config.KNN_NEIGHBOURS, weights="distance")


def _svm():
    return SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
               random_state=config.RANDOM_STATE)


def build_pipelines(top_k=config.TOP_K_FEATURES, pca_var=config.PCA_VARIANCE):
    """Return {name: unfitted Pipeline}."""
    def steps(*tail):
        return Pipeline([("scale", StandardScaler()),
                         ("fisher", FisherScoreSelector(top_k)),
                         *tail])

    return {
        "PCA + kNN":       steps(("pca", PCA(pca_var, random_state=config.RANDOM_STATE)), ("clf", _knn())),
        "LDA + kNN":       steps(("lda", LinearDiscriminantAnalysis()), ("clf", _knn())),
        "PCA + LDA + kNN": steps(("pca", PCA(pca_var, random_state=config.RANDOM_STATE)),
                                 ("lda", LinearDiscriminantAnalysis()), ("clf", _knn())),
        "PCA + SVM":       steps(("pca", PCA(pca_var, random_state=config.RANDOM_STATE)), ("clf", _svm())),
        "LDA + SVM":       steps(("lda", LinearDiscriminantAnalysis()), ("clf", _svm())),
    }


def pca_knn_with(n_components, top_k=config.TOP_K_FEATURES):
    """PCA + kNN with a fixed number of components (for the accuracy-vs-components curve)."""
    return Pipeline([("scale", StandardScaler()),
                     ("fisher", FisherScoreSelector(top_k)),
                     ("pca", PCA(n_components, random_state=config.RANDOM_STATE)),
                     ("clf", _knn())])


def lda_knn_with_top_k(top_k):
    """LDA + kNN on the top-k Fisher features (for the accuracy-vs-k curve)."""
    return Pipeline([("scale", StandardScaler()),
                     ("fisher", FisherScoreSelector(top_k)),
                     ("lda", LinearDiscriminantAnalysis()),
                     ("clf", _knn())])

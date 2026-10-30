"""
run_solution2.py
================
SOLUTION 2 : Handcrafted features + Fisher-score selection + PCA / LDA
             + classical classifiers (kNN, SVM)

Pipeline
    image -> 90 colour & texture features (colour moments, HSV histograms,
             GLCM, LBP)
          -> standardise
          -> FILTER feature selection: keep the top-K by Fisher score
          -> subspace projection: PCA, LDA or PCA->LDA
          -> kNN or SVM
    Five such pipelines are compared by 5-fold cross-validation on the
    training set; the winner is evaluated once on the test set.

Syllabus coverage: Unit I (feature vectors, pattern space), Unit III (PCA,
LDA, dimensionality reduction, Fisher score, information gain, filter
selection), Unit II (performance evaluation).

Usage
    python run_solution2.py
    python run_solution2.py --data-dir path/to/folder --robustness
"""

import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import config
from common.evaluation import compute_metrics, full_evaluation, print_metrics, save_results
from common.experiment import base_arg_parser, cached_features, load_and_split, load_field_set
from common.features import HANDCRAFTED_FEATURE_NAMES, extract_batch, handcrafted_features
from common.robustness import all_perturbations
from solution2_subspace_classical import plots
from solution2_subspace_classical.feature_selection import fisher_score, information_gain
from solution2_subspace_classical.pipelines import (build_pipelines, lda_knn_with_top_k,
                                                    pca_knn_with)

SOLUTION = "solution2"
TITLE = "Fisher score + PCA/LDA + kNN/SVM"


def main():
    parser = base_arg_parser("Solution 2: handcrafted features + PCA/LDA + kNN/SVM")
    parser.add_argument("--top-k", type=int, default=config.TOP_K_FEATURES,
                        help="number of features kept by the Fisher-score filter")
    args = parser.parse_args()
    out = Path(args.results_dir) / SOLUTION
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 1. data
    images, labels, class_names, tr, te = load_and_split(args)
    X, feat_sec = cached_features(images, handcrafted_features, "handcrafted", args)
    Xtr, Xte, ytr, yte = X[tr], X[te], labels[tr], labels[te]
    top_k = min(args.top_k, X.shape[1])

    # ---------------------------------- 2. feature evaluation criteria plot
    Xtr_std = StandardScaler().fit_transform(Xtr)
    fs = fisher_score(Xtr_std, ytr)
    ig = information_gain(Xtr_std, ytr)
    plots.feature_ranking(fs, ig, HANDCRAFTED_FEATURE_NAMES, out / "feature_ranking.png")

    # -------------------------- 3. model selection by cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    pipelines = build_pipelines(top_k)
    cv_results = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="accuracy")
        cv_results[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"[cv] {name:<18s} {scores.mean():.4f} +/- {scores.std():.4f}")
    plots.cv_comparison(cv_results, out / "cv_model_selection.png")
    best_name = max(cv_results, key=lambda n: cv_results[n]["mean"])
    print(f"[solution2] selected pipeline: {best_name}")

    # ------------------------------------- 4. train the winner, evaluate
    model = clone(pipelines[best_name])
    t0 = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t0
    t0 = time.time()
    pred = model.predict(Xte)
    pred_sec = (time.time() - t0) / len(Xte)
    proba = model.predict_proba(Xte)
    tag = best_name.replace(" + ", "_").replace(" ", "").lower()
    metrics = full_evaluation(tag, yte, pred, proba, class_names, out)
    print_metrics(best_name, metrics)

    # Also report every pipeline on the test set (for the paper's table)
    all_test = {}
    for name, pipe in pipelines.items():
        p = clone(pipe).fit(Xtr, ytr).predict(Xte)
        all_test[name] = compute_metrics(yte, p, class_names=class_names)

    # ---------------------------------------- 5. PCA / LDA visualisations
    scale = StandardScaler().fit(Xtr)
    sel = np.sort(np.argsort(fisher_score(scale.transform(Xtr), ytr))[::-1][:top_k])
    Ztr = scale.transform(Xtr)[:, sel]
    plots.pca_variance(PCA(random_state=config.RANDOM_STATE).fit(Ztr), out / "pca_explained_variance.png")
    Z_pca = PCA(2, random_state=config.RANDOM_STATE).fit_transform(Ztr)
    Z_lda = LinearDiscriminantAnalysis(n_components=min(2, len(class_names) - 1)).fit_transform(Ztr, ytr)
    plots.projection_scatter(Z_pca, Z_lda, ytr, class_names, out / "pca_vs_lda_projection.png")

    # Accuracy vs. number of PCA components, and vs. number of selected features
    max_comp = min(top_k, len(tr) - 1)
    comps = sorted({c for c in (2, 3, 5, 8, 10, 15, 20, 30, 40, 60) if c <= max_comp})
    acc_c = [compute_metrics(yte, pca_knn_with(c, top_k).fit(Xtr, ytr).predict(Xte))["accuracy"] for c in comps]
    plots.curve(comps, acc_c, "Number of PCA components", "PCA + kNN: accuracy vs. components",
                out / "accuracy_vs_pca_components.png")
    ks = sorted({k for k in (5, 10, 20, 30, 45, 60, 75, 90) if k <= X.shape[1]})
    acc_k = [compute_metrics(yte, lda_knn_with_top_k(k).fit(Xtr, ytr).predict(Xte))["accuracy"] for k in ks]
    plots.curve(ks, acc_k, "Top-K features kept (Fisher score)", "LDA + kNN: accuracy vs. features kept",
                out / "accuracy_vs_top_k_features.png")

    # ------------------------------------------------ 6. robustness tests
    robustness = {}
    if args.robustness:
        print("[solution2] robustness: re-extracting features from degraded test images ...")
        for kind, degraded in all_perturbations(images[te]):
            Xd, _ = extract_batch(degraded, handcrafted_features, desc=kind, verbose=False)
            robustness[kind] = compute_metrics(yte, model.predict(Xd), class_names=class_names)
            print_metrics(f"robustness/{kind}", robustness[kind])

    field = load_field_set(args, class_names)
    field_metrics = None
    if field is not None:
        Xf, _ = extract_batch(field[0], handcrafted_features, desc="field", verbose=False)
        field_metrics = full_evaluation(f"field_{tag}", field[1], model.predict(Xf),
                                        model.predict_proba(Xf), class_names, out)
        print_metrics("field photos", field_metrics)

    # --------------------------------------------------------- 7. save
    save_results({
        "solution": SOLUTION,
        "title": TITLE,
        "main_method": best_name,
        "class_names": class_names,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_features": int(X.shape[1]), "n_selected_features": int(top_k),
        "metrics": metrics,
        "cv_model_selection": cv_results,
        "variants": all_test,
        "train_time_s": train_time,
        "feature_ms_per_image": feat_sec * 1000,
        "predict_ms_per_image": (feat_sec + pred_sec) * 1000,
        "robustness": robustness,
        "field": field_metrics,
    }, out / "results.json")
    print(f"[solution2] all outputs written to {out}")


if __name__ == "__main__":
    main()

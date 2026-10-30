"""
run_solution1.py
================
SOLUTION 1 : K-means lesion segmentation  +  Gaussian Bayes classifier

Pipeline
    image -> leaf mask (Otsu on saturation)
          -> k-means colour clustering inside the leaf (UNSUPERVISED)
          -> lesion features (area, spot count, spot colour, ...)
          -> standardise
          -> Gaussian Bayes classifier (from scratch), three decision rules:
               a) full covariance, minimum-error (MAP)      <- main result
               b) diagonal covariance  = Naive Bayes
               c) full covariance, minimum-RISK (missing a disease costs more)

Syllabus coverage: Unit II (Bayes theorem, discriminant functions, minimum
error / minimum risk, error estimation) and Unit IV (k-means clustering,
image segmentation).

Usage
    python run_solution1.py                       # uses data/tomato
    python run_solution1.py --data-dir path/to/folder --robustness
"""

import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import config
from common.dataset import find_healthy_class
from common.evaluation import compute_metrics, full_evaluation, print_metrics, save_results
from common.experiment import base_arg_parser, cached_features, load_and_split, load_field_set
from common.features import extract_batch, lesion_features
from common.robustness import all_perturbations
from solution1_clustering_bayes.bayes_classifier import (GaussianBayesClassifier,
                                                         miss_cost_loss_matrix)
from solution1_clustering_bayes.plots import (feature_distributions, risk_tradeoff,
                                              segmentation_examples)

SOLUTION = "solution1"
TITLE = "K-means segmentation + Bayes"


def main():
    parser = base_arg_parser("Solution 1: k-means lesion segmentation + Gaussian Bayes classifier")
    parser.add_argument("--miss-cost", type=float, default=config.MISS_COST,
                        help="loss of calling a diseased leaf healthy (minimum-risk rule)")
    args = parser.parse_args()
    out = Path(args.results_dir) / SOLUTION
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 1. data
    images, labels, class_names, tr, te = load_and_split(args)
    healthy = find_healthy_class(class_names)

    # ------------------------------------- 2. segmentation figure + features
    print("[solution1] drawing segmentation examples ...")
    segmentation_examples(images[te], labels[te], class_names, out / "segmentation_examples.png")

    X, feat_sec = cached_features(images, lesion_features, "lesion", args)
    feature_distributions(X, labels, class_names, out / "lesion_feature_distributions.png")

    # Standardise with TRAINING statistics only (no information leaks from test)
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
    ytr, yte = labels[tr], labels[te]

    # ------------------------------------------------------ 3. train models
    t0 = time.time()
    bayes = GaussianBayesClassifier(naive=False, reg=config.BAYES_REG).fit(Xtr, ytr)
    train_time = time.time() - t0
    naive = GaussianBayesClassifier(naive=True, reg=config.BAYES_REG).fit(Xtr, ytr)

    # --------------------------------------------------------- 4. evaluate
    t0 = time.time()
    pred = bayes.predict(Xte)
    pred_sec = (time.time() - t0) / len(Xte)
    proba = bayes.predict_proba(Xte)
    metrics = full_evaluation("bayes_full_cov", yte, pred, proba, class_names, out)
    print_metrics("Bayes full covariance (MAP)", metrics)

    naive_metrics = full_evaluation("naive_bayes", yte, naive.predict(Xte),
                                    naive.predict_proba(Xte), class_names, out)
    print_metrics("Naive Bayes", naive_metrics)

    # Minimum-risk rule and the accuracy / miss-rate trade-off curve
    loss = miss_cost_loss_matrix(len(class_names), healthy, args.miss_cost)
    risk_pred = bayes.predict_min_risk(Xte, loss)
    risk_metrics = full_evaluation("bayes_min_risk", yte, risk_pred, proba, class_names, out)
    print_metrics(f"Bayes minimum risk (miss cost {args.miss_cost})", risk_metrics)

    if healthy is not None:
        costs = [1, 2, 5, 10, 20, 50, 100]
        accs, misses = [], []
        for c in costs:
            p = bayes.predict_min_risk(Xte, miss_cost_loss_matrix(len(class_names), healthy, c))
            m = compute_metrics(yte, p, class_names=class_names)
            accs.append(m["accuracy"])
            misses.append(m.get("miss_rate", 0.0))
        risk_tradeoff(costs, accs, misses, out / "risk_tradeoff.png")

    # ------------------------------------------------ 5. robustness tests
    robustness = {}
    if args.robustness:
        print("[solution1] robustness: re-extracting features from degraded test images ...")
        for kind, degraded in all_perturbations(images[te]):
            Xd, _ = extract_batch(degraded, lesion_features, desc=kind, verbose=False)
            m = compute_metrics(yte, bayes.predict(scaler.transform(Xd)), class_names=class_names)
            robustness[kind] = m
            print_metrics(f"robustness/{kind}", m)

    field = load_field_set(args, class_names)
    field_metrics = None
    if field is not None:
        Xf, _ = extract_batch(field[0], lesion_features, desc="field", verbose=False)
        pf = bayes.predict(scaler.transform(Xf))
        field_metrics = full_evaluation("field_bayes_full_cov", field[1], pf,
                                        bayes.predict_proba(scaler.transform(Xf)), class_names, out)
        print_metrics("field photos", field_metrics)

    # --------------------------------------------------------- 6. save
    save_results({
        "solution": SOLUTION,
        "title": TITLE,
        "main_method": "Gaussian Bayes, full covariance, MAP rule",
        "class_names": class_names,
        "n_train": int(len(tr)), "n_test": int(len(te)), "n_features": int(X.shape[1]),
        "metrics": metrics,
        "variants": {"naive_bayes": naive_metrics, "min_risk": risk_metrics},
        "train_time_s": train_time,
        "feature_ms_per_image": feat_sec * 1000,
        "predict_ms_per_image": (feat_sec + pred_sec) * 1000,
        "robustness": robustness,
        "field": field_metrics,
    }, out / "results.json")
    print(f"[solution1] all outputs written to {out}")


if __name__ == "__main__":
    main()

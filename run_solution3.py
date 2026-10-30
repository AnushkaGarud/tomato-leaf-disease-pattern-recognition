"""
run_solution3.py
================
SOLUTION 3 : Artificial neural networks

Models
    a) Perceptron (from scratch)                 - linear baseline
    b) Multilayer Perceptron + backpropagation   - MAIN RESULT, from scratch
       on the same 90 handcrafted features as Solution 2
    c) Small CNN on raw pixels (optional, --cnn) - learns its own features;
       needs Keras 3 + a backend (TensorFlow, PyTorch or JAX)

Syllabus coverage: Unit V (ANNs, perceptron, MLP, backpropagation,
training and testing, applications in pattern recognition).

Usage
    python run_solution3.py                       # perceptron + MLP
    python run_solution3.py --cnn                 # + CNN benchmark
    python run_solution3.py --cnn --robustness
"""

import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import config
from common.evaluation import compute_metrics, full_evaluation, print_metrics, save_results
from common.experiment import base_arg_parser, cached_features, load_and_split, load_field_set
from common.features import extract_batch, handcrafted_features
from common.robustness import all_perturbations
from solution3_neural import plots
from solution3_neural.cnn_keras import CNNClassifier, keras_available
from solution3_neural.mlp_numpy import MLP, Perceptron

SOLUTION = "solution3"
TITLE = "Neural network (MLP, backprop)"


def main():
    parser = base_arg_parser("Solution 3: perceptron, MLP with backpropagation and optional CNN")
    parser.add_argument("--cnn", action="store_true", help="also train the CNN on raw pixels (needs Keras)")
    parser.add_argument("--cnn-size", type=int, default=96, help="CNN input size in pixels")
    parser.add_argument("--epochs", type=int, default=config.MLP_EPOCHS, help="max MLP epochs")
    args = parser.parse_args()
    out = Path(args.results_dir) / SOLUTION
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 1. data
    images, labels, class_names, tr, te = load_and_split(args)
    X, feat_sec = cached_features(images, handcrafted_features, "handcrafted", args)
    scaler = StandardScaler().fit(X[tr])                 # neural nets need scaled inputs
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
    ytr, yte = labels[tr], labels[te]
    test_summary = {}

    # ---------------------------------------------------- 2. perceptron
    perc = Perceptron().fit(Xtr, ytr)
    perc_metrics = full_evaluation("perceptron", yte, perc.predict(Xte), perc.predict_proba(Xte),
                                   class_names, out)
    print_metrics("Perceptron", perc_metrics)
    test_summary["Perceptron"] = perc_metrics

    # ----------------------------------------------- 3. MLP (main model)
    print(f"[solution3] training MLP {X.shape[1]} -> {config.MLP_HIDDEN} -> {len(class_names)}")
    mlp = MLP(epochs=args.epochs)
    t0 = time.time()
    mlp.fit(Xtr, ytr)
    train_time = time.time() - t0
    t0 = time.time()
    pred = mlp.predict(Xte)
    pred_sec = (time.time() - t0) / len(Xte)
    metrics = full_evaluation("mlp", yte, pred, mlp.predict_proba(Xte), class_names, out)
    print_metrics("MLP (backprop)", metrics)
    test_summary["MLP"] = metrics
    plots.training_curves(mlp.history_, out / "mlp_training_curves.png",
                          title="MLP trained with backpropagation", best_epoch=mlp.best_epoch_)
    plots.hidden_representation(mlp, Xte, yte, class_names, out / "mlp_hidden_representation.png")

    # ------------------------------------------------- 4. optional CNN
    cnn, cnn_result = None, None
    if args.cnn:
        if not keras_available():
            print("[solution3] --cnn skipped: install Keras 3 and a backend, e.g. "
                  "'pip install keras tensorflow'")
        else:
            print(f"[solution3] training CNN on {args.cnn_size}x{args.cnn_size} images ...")
            cnn = CNNClassifier(len(class_names), size=args.cnn_size)
            t0 = time.time()
            cnn.fit(images[tr], ytr)
            cnn_train = time.time() - t0
            t0 = time.time()
            cnn_proba = cnn.predict_proba(images[te])
            cnn_pred_sec = (time.time() - t0) / len(te)
            cnn_metrics = full_evaluation("cnn", yte, cnn_proba.argmax(1), cnn_proba, class_names, out)
            print_metrics("CNN", cnn_metrics)
            test_summary["CNN"] = cnn_metrics
            plots.training_curves(cnn.history_, out / "cnn_training_curves.png",
                                  title="CNN trained end-to-end on pixels")
            cnn_result = {"metrics": cnn_metrics, "train_time_s": cnn_train,
                          "predict_ms_per_image": cnn_pred_sec * 1000, "n_params": cnn.n_params()}

    plots.model_comparison(test_summary, out / "neural_model_comparison.png")

    # ------------------------------------------------ 5. robustness tests
    robustness, cnn_robustness = {}, {}
    if args.robustness:
        print("[solution3] robustness: re-extracting features from degraded test images ...")
        for kind, degraded in all_perturbations(images[te]):
            Xd, _ = extract_batch(degraded, handcrafted_features, desc=kind, verbose=False)
            robustness[kind] = compute_metrics(yte, mlp.predict(scaler.transform(Xd)), class_names=class_names)
            print_metrics(f"robustness/MLP/{kind}", robustness[kind])
            if cnn is not None:
                cnn_robustness[kind] = compute_metrics(yte, cnn.predict(degraded), class_names=class_names)
                print_metrics(f"robustness/CNN/{kind}", cnn_robustness[kind])
        if cnn_result:
            cnn_result["robustness"] = cnn_robustness

    field = load_field_set(args, class_names)
    field_metrics = None
    if field is not None:
        Xf, _ = extract_batch(field[0], handcrafted_features, desc="field", verbose=False)
        Xf = scaler.transform(Xf)
        field_metrics = full_evaluation("field_mlp", field[1], mlp.predict(Xf), mlp.predict_proba(Xf),
                                        class_names, out)
        print_metrics("field photos (MLP)", field_metrics)
        if cnn is not None:
            cp = cnn.predict_proba(field[0])
            cnn_result["field"] = full_evaluation("field_cnn", field[1], cp.argmax(1), cp, class_names, out)
            print_metrics("field photos (CNN)", cnn_result["field"])

    # --------------------------------------------------------- 6. save
    save_results({
        "solution": SOLUTION,
        "title": TITLE,
        "main_method": f"MLP {X.shape[1]}-{'-'.join(map(str, config.MLP_HIDDEN))}-{len(class_names)}, "
                       "SGD + momentum, backpropagation",
        "class_names": class_names,
        "n_train": int(len(tr)), "n_test": int(len(te)), "n_features": int(X.shape[1]),
        "metrics": metrics,
        "variants": {"perceptron": perc_metrics},
        "best_epoch": int(mlp.best_epoch_),
        "train_time_s": train_time,
        "feature_ms_per_image": feat_sec * 1000,
        "predict_ms_per_image": (feat_sec + pred_sec) * 1000,
        "robustness": robustness,
        "field": field_metrics,
        "cnn": cnn_result,
    }, out / "results.json")
    print(f"[solution3] all outputs written to {out}")


if __name__ == "__main__":
    main()

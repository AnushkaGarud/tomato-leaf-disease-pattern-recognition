# Tomato Leaf Disease Detection: Three Pattern Recognition Solutions

One problem, three solutions, compared on the same data and split.

| | Solution | Script |
|---|---|---|
| 1 | K-means lesion segmentation + Gaussian Bayes classifier | `run_solution1.py` |
| 2 | Handcrafted features + Fisher score + PCA/LDA + kNN/SVM | `run_solution2.py` |
| 3 | Perceptron + MLP with backpropagation (+ optional CNN) | `run_solution3.py` |

## Quick start

```bash
pip install -r requirements.txt

# 1. smoke test on synthetic leaves (2-4 minutes)
python scripts/make_synthetic_data.py
python run_all.py --data-dir data/synthetic --robustness

# 2. real experiment: put PlantVillage tomato folders in data/tomato/
python run_all.py --robustness            # add --cnn for the CNN benchmark
```

Each script also runs on its own, for example `python run_solution2.py --robustness`.
Run `python run_solution1.py --help` to see all options.

Outputs go to `results/solution1`, `results/solution2`, `results/solution3`
and `results/comparison` (figures as PNG, tables as CSV/Markdown, metrics as JSON).

Backpropagation check: `python -m solution3_neural.mlp_numpy`

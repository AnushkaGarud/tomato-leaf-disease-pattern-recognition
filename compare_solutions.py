"""
compare_solutions.py
====================
Collect the results.json files written by the three run scripts and draw the
side-by-side figures and tables for the research paper.

Run AFTER run_solution1.py, run_solution2.py and run_solution3.py:
    python compare_solutions.py

Outputs (results/comparison/)
    metrics_comparison.png     accuracy / precision / recall / F1 per solution
    speed_vs_accuracy.png      inference time per image vs. accuracy
    robustness.png             accuracy on clean vs. degraded test images
    robustness_drop.png        accuracy lost under each degradation
    miss_rate.png              share of diseased leaves called healthy
    lab_vs_field.png           (only if a field dataset was evaluated)
    summary.csv / summary.md   one table with every number above
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
METRICS = [("accuracy", "Accuracy"), ("precision_macro", "Precision"),
           ("recall_macro", "Recall"), ("f1_macro", "F1 score")]


def load_entries(results_dir):
    """Return a list of dicts, one per method to compare (CNN added if it was trained)."""
    entries = []
    for i, sol in enumerate(("solution1", "solution2", "solution3"), start=1):
        path = Path(results_dir) / sol / "results.json"
        if not path.exists():
            print(f"[compare] missing {path} - run run_{sol}.py first")
            continue
        r = json.loads(path.read_text())
        entries.append({
            "name": f"S{i}: {r['title']}", "short": f"S{i}", "method": r["main_method"],
            "metrics": r["metrics"], "predict_ms": r["predict_ms_per_image"],
            "train_s": r["train_time_s"], "robustness": r.get("robustness") or {},
            "field": r.get("field"),
        })
        cnn = r.get("cnn")
        if cnn:
            entries.append({
                "name": f"S{i}b: CNN on raw pixels", "short": f"S{i}b",
                "method": f"CNN, {cnn['n_params']:,} parameters", "metrics": cnn["metrics"],
                "predict_ms": cnn["predict_ms_per_image"], "train_s": cnn["train_time_s"],
                "robustness": cnn.get("robustness") or {}, "field": cnn.get("field"),
            })
    return entries


def metrics_comparison(entries, path):
    x = np.arange(len(METRICS))
    w = 0.8 / len(entries)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for j, e in enumerate(entries):
        vals = [e["metrics"][k] for k, _ in METRICS]
        bars = ax.bar(x + (j - (len(entries) - 1) / 2) * w, vals, w, label=e["name"], color=COLORS[j % 4])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x, [label for _, label in METRICS])
    lo = min(e["metrics"][k] for e in entries for k, _ in METRICS)
    ax.set_ylim(max(0, lo - 0.1), 1.06)
    ax.set_ylabel("Score on the test set (macro average)")
    ax.set_title("Overall performance of the three solutions")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def speed_vs_accuracy(entries, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    times = [e["predict_ms"] for e in entries]
    for j, e in enumerate(entries):
        ax.scatter(e["predict_ms"], e["metrics"]["accuracy"], s=90, color=COLORS[j % 4],
                   label=e["name"], alpha=0.85, zorder=3)
        # stagger the labels vertically so overlapping points stay readable
        ax.annotate(e["short"], (e["predict_ms"], e["metrics"]["accuracy"]),
                    textcoords="offset points", xytext=(8, 10 - 9 * j), fontsize=8)
    log = max(times) / max(min(times), 1e-9) > 20       # log axis only for a wide range
    if log:
        ax.set_xscale("log")
    ax.set_xlabel("Inference time per image incl. feature extraction (ms"
                  + (", log scale)" if log else ")"))
    ax.set_ylabel("Test accuracy")
    ax.set_title("Speed vs. accuracy (top-left is best)")
    ax.margins(x=0.25, y=0.15)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def robustness_plots(entries, path_abs, path_drop):
    kinds = list(config.PERTURBATIONS)
    with_rob = [e for e in entries if e["robustness"]]
    if not with_rob:
        print("[compare] no robustness results (run the solutions with --robustness)")
        return
    labels = ["clean"] + [k.replace("_", " ") for k in kinds]
    x = np.arange(len(labels))
    w = 0.8 / len(with_rob)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for j, e in enumerate(with_rob):
        vals = [e["metrics"]["accuracy"]] + [e["robustness"].get(k, {}).get("accuracy", np.nan) for k in kinds]
        ax.bar(x + (j - (len(with_rob) - 1) / 2) * w, vals, w, label=e["name"], color=COLORS[entries.index(e) % 4])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Robustness to poor photo conditions")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path_abs, bbox_inches="tight")
    plt.close(fig)

    x = np.arange(len(kinds))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for j, e in enumerate(with_rob):
        drop = [e["metrics"]["accuracy"] - e["robustness"].get(k, {}).get("accuracy", np.nan) for k in kinds]
        ax.bar(x + (j - (len(with_rob) - 1) / 2) * w, drop, w, label=e["name"], color=COLORS[entries.index(e) % 4])
    ax.set_xticks(x, [k.replace("_", " ") for k in kinds])
    ax.set_ylabel("Accuracy lost (clean - degraded)")
    ax.set_title("Accuracy drop under each degradation (lower is better)")
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path_drop, bbox_inches="tight")
    plt.close(fig)


def miss_rate_plot(entries, path):
    vals = [e["metrics"].get("miss_rate") for e in entries]
    if all(v is None for v in vals):
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar([e["short"] for e in entries], [v or 0 for v in vals], color=[COLORS[j % 4] for j in range(len(entries))])
    for j, v in enumerate(vals):
        ax.text(j, (v or 0) + 0.002, f"{(v or 0):.3f}", ha="center", fontsize=7)
    ax.set_ylabel("Miss rate")
    ax.set_title("Diseased leaves wrongly called healthy (lower is better)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def lab_vs_field(entries, path):
    with_field = [e for e in entries if e["field"]]
    if not with_field:
        return
    x = np.arange(len(with_field))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(x - 0.18, [e["metrics"]["accuracy"] for e in with_field], 0.36, label="Lab test set", color="#4C72B0")
    ax.bar(x + 0.18, [e["field"]["accuracy"] for e in with_field], 0.36, label="Field photos", color="#C44E52")
    ax.set_xticks(x, [e["short"] for e in with_field])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Laboratory vs. field conditions")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_tables(entries, out):
    kinds = list(config.PERTURBATIONS)
    header = (["Solution", "Method"] + [label for _, label in METRICS] + ["ROC-AUC", "Miss rate",
              "Train time (s)", "Inference (ms/img)"] + [f"Acc. {k}" for k in kinds] + ["Acc. field"])
    rows = []
    for e in entries:
        m = e["metrics"]
        row = [e["name"], e["method"]] + [f"{m[k]:.4f}" for k, _ in METRICS]
        row += [f"{m['roc_auc_macro']:.4f}" if "roc_auc_macro" in m else "",
                f"{m['miss_rate']:.4f}" if "miss_rate" in m else "",
                f"{e['train_s']:.2f}", f"{e['predict_ms']:.2f}"]
        row += [f"{e['robustness'][k]['accuracy']:.4f}" if k in e["robustness"] else "" for k in kinds]
        row += [f"{e['field']['accuracy']:.4f}" if e["field"] else ""]
        rows.append(row)

    with open(out / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description="Compare the three solutions")
    ap.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    args = ap.parse_args()
    entries = load_entries(args.results_dir)
    if not entries:
        return
    out = Path(args.results_dir) / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    metrics_comparison(entries, out / "metrics_comparison.png")
    speed_vs_accuracy(entries, out / "speed_vs_accuracy.png")
    robustness_plots(entries, out / "robustness.png", out / "robustness_drop.png")
    miss_rate_plot(entries, out / "miss_rate.png")
    lab_vs_field(entries, out / "lab_vs_field.png")
    write_tables(entries, out)
    print(f"\n[compare] figures and tables written to {out}")


if __name__ == "__main__":
    main()

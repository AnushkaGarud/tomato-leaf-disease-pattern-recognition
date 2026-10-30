"""
run_all.py
==========
Run the three solutions one after another and then the comparison.
Any extra arguments are passed to every solution, e.g.

    python run_all.py --robustness
    python run_all.py --data-dir data/synthetic --robustness
    python run_all.py --robustness --cnn          (--cnn only goes to Solution 3)
"""

import subprocess
import sys

args = sys.argv[1:]
cnn = "--cnn" in args
common = [a for a in args if a != "--cnn"]

steps = [
    ["run_solution1.py", *common],
    ["run_solution2.py", *common],
    ["run_solution3.py", *common, *(["--cnn"] if cnn else [])],
]
for step in steps:
    print(f"\n{'=' * 70}\n>>> python {' '.join(step)}\n{'=' * 70}")
    subprocess.run([sys.executable, *step], check=True)

# the comparison only needs to know where the results are
rd = []
if "--results-dir" in common:
    i = common.index("--results-dir")
    rd = common[i:i + 2]
print(f"\n{'=' * 70}\n>>> python compare_solutions.py\n{'=' * 70}")
subprocess.run([sys.executable, "compare_solutions.py", *rd], check=True)

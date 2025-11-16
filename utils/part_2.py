"""
Utilities for Part 2 analysis: plotting unlearning results.

This module only adds plotting utilities on top of the Part 1 helpers.
"""

import os

import matplotlib.pyplot as plt

from utils.part_1 import ensure_dir

def plot_ifeval_inst_strict_by_model(labels, inst_values, out_png):
    plt.rcParams["font.family"] = "serif"
    width = max(6.0, 1.5 * max(1, len(labels)))
    plt.figure(figsize=(width, 4.5))

    xs = list(range(len(labels)))
    plt.bar(xs, inst_values)

    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.ylabel("IFEval instruction-level strict accuracy")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=220)
    plt.close()
    print("[info] Saved IFEval inst-strict plot to: %s" % out_png)

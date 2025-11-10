from __future__ import annotations

import csv
import json
import os

import matplotlib.pyplot as plt

def ensure_dir(p):
    if not p:
        return
    os.makedirs(p, exist_ok=True)

def sanitize_filename(s):
    return s.replace("/", "__").replace(":", "_")

def plot_memorization(sizes, trait2acc, out_png):
    plt.rcParams["font.family"] = "serif"
    colors = ["#aec6cf", "#cdb4db", "#bde0fe", "#c3f0ca"]
    plt.figure(figsize=(7.2, 4.5))

    for i, (label, series) in enumerate(trait2acc.items()):
        plt.plot(
            sizes,
            series,
            marker="o",
            linewidth=2.0,
            color=colors[i % len(colors)],
            label=label,
        )
        
    plt.xscale("log")
    plt.xlabel("Parameters (in billions)")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.xticks(list(sizes), [str(s) for s in sizes])
    plt.legend(frameon=False)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=220)
    print(f"[info] Saved memorization plot to: {out_png}")

def save_table(rows, out_csv, out_json):
    if not rows:
        return
    ensure_dir(os.path.dirname(out_csv) or ".")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(out_json, "w") as f:
        json.dump(list(rows), f, indent=2)
    print(f"[info] Saved results to: {out_csv} and {out_json}")

def open_jsonl(path):
    if not path:
        return None
    ensure_dir(os.path.dirname(path) or ".")
    return open(path, "w")

def write_jsonl(f, record):
    if f is None:
        return
    f.write(json.dumps(record) + "\n")

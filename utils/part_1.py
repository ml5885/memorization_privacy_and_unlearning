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
    plt.figure(figsize=(7.2, 4.5))

    fox = ["#FFC907", "#C62027", "#817F3E", "#9F5831"]

    for i, (label, series) in enumerate(trait2acc.items()):
        color = fox[i % len(fox)]
        plt.plot(
            sizes,
            series,
            marker="o",
            linewidth=2.0,
            label=label,
            color=color,
        )

    plt.xlabel("Parameters (in billions)")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.xticks(list(sizes), [str(s) for s in sizes])
    plt.legend(frameon=False)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=220)
    plt.close()
    print("[info] Saved memorization plot to: %s" % out_png)

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
    print("[info] Saved results to: %s and %s" % (out_csv, out_json))

def open_jsonl(path):
    if not path:
        return None
    ensure_dir(os.path.dirname(path) or ".")
    return open(path, "w")

def write_jsonl(f, record):
    if f is None:
        return
    f.write(json.dumps(record) + "\n")

def get_model_label(model_str):
    if model_str == "google/gemma-3-4b-it":
        return "Gemma-3-4b-it"
    elif "dpo_" in model_str and "gemma-3-4b-it" in model_str:
        return "Gemma-3-4b-it (DPO)"
    elif "rmu_" in model_str and "gemma-3-4b-it" in model_str:
        return "Gemma-3-4b-it (RMU)"
    else:
        return model_str
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt

from utils.part_1 import ensure_dir

def default_refusal():
    return "I cannot provide this information."

def split_forget_holdout(items, held_out_trait):
    forget = [r for r in items if r["trait"] in {"Type 1", "HP", "Defense"}]
    holdout = [r for r in items if r["trait"] == held_out_trait]
    return forget, holdout

def make_forget_pairs(records, refusal=None):
    if refusal is None:
        refusal = default_refusal()
    
    pairs = []
    
    for r in records:
        if r["format"] == "mcq":
            prompt = r["prompt"]
            rejected = r["answer_letter"]
            chosen = refusal
        else:
            prompt = r["prompt"]
            rejected = r["answer"]
            chosen = refusal
        
        pairs.append((prompt, chosen, rejected))
    
    return pairs

def pack_result_row(name, size_b, trait_acc, tqa, ife, trait_labels):
    row = {"model": name, "size_b": size_b}
    
    for k, label in trait_labels.items():
        row[label] = round(trait_acc.get(k, 0.0), 4)
    
    row["TriviaQA"] = round(tqa, 4)
    row["IFEval_prompt_strict"] = round(ife["prompt_strict"], 4)
    row["IFEval_inst_strict"] = round(ife["inst_strict"], 4)
    row["IFEval_prompt_loose"] = round(ife["prompt_loose"], 4)
    row["IFEval_inst_loose"] = round(ife["inst_loose"], 4)
    
    return row

def plot_grouped(results, out_png):
    plt.rcParams["font.family"] = "serif"
    
    labels = ["Forget Traits", "Held-out Trait", "General Capability"]
    keys = ["forget", "heldout", "general"]
    methods = list(results.keys())
    
    x = range(len(labels))
    w = 0.8 / max(1, len(methods))
    
    fox = ["#FFC907","#C62027","#817F3E","#9F5831"]
    colors = fox[:len(methods)]
    
    plt.figure(figsize=(7.5, 4.5))
    for i, m in enumerate(methods):
        vals = [results[m][k] for k in keys]
        xs = [xx + (i - (len(methods)-1)/2)*w for xx in x]
        bars = plt.bar(xs, vals, width=w, label=m, color=colors[i])
        for bar, val in zip(bars, vals):
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.01, 
                f"{val:.2f}", 
                ha='center', va='bottom', fontsize=10
            )
    
    plt.ylim(0.0, 1.0)
    plt.xticks(list(x), labels)
    plt.ylabel("Accuracy")
    plt.legend(frameon=False)
    
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[info] Saved grouped plot to: {out_png}")

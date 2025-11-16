from __future__ import annotations

import argparse
import json
import os
from typing import List

from data.pokemon import build_pokemon_benchmark
from unlearning.dpo import run_dpo_unlearning
from unlearning.rmu import run_rmu_unlearning
from utils.part_1 import ensure_dir, sanitize_filename, save_table
from utils.part_2 import plot_ifeval_inst_strict_by_model


FORGET_TRAITS = ["Type 1", "HP", "Defense"]
RETAIN_TRAITS = ["Speed"]

def run_training(args):
    ensure_dir(args.outdir)

    bench_path = args.pokemon_bench
    if not os.path.exists(bench_path):
        print(f"[part2] Pokemon MCQ benchmark not found at {bench_path}, building it...")
        ensure_dir(os.path.dirname(bench_path) or ".")
        build_pokemon_benchmark(
            raw_csv_path=args.pokemon_csv,
            out_csv_path=bench_path,
            use_mcq=True,
            min_per_trait=args.min_per_trait,
        )
        print("[part2] Pokemon MCQ benchmark created.")

    algo = args.algo.lower()
    if algo not in {"dpo", "rmu"}:
        raise ValueError(f"Unknown algo: {algo}")

    model_tag = sanitize_filename(args.model)
    run_tag = f"{algo}_pokemon_{model_tag}"
    run_dir = os.path.join(args.outdir, run_tag)
    ensure_dir(run_dir)

    print(f"[part2] Running {algo.upper()} unlearning for model={args.model}")
    print(f"[part2] Forget traits: {FORGET_TRAITS}, retain traits: {RETAIN_TRAITS}")
    print(f"[part2] Output directory: {run_dir}")

    if algo == "dpo":
        adapter_dir = run_dpo_unlearning(
            model_id=args.model,
            pokemon_bench_path=bench_path,
            outdir=run_dir,
            forget_traits=FORGET_TRAITS,
            lr=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            beta=args.beta,
            local_files_only=args.local_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    else:
        adapter_dir = run_rmu_unlearning(
            model_id=args.model,
            pokemon_bench_path=bench_path,
            outdir=run_dir,
            forget_traits=FORGET_TRAITS,
            retain_traits=RETAIN_TRAITS,
            lr=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            layer_index=args.layer_index,
            c=args.rmu_c,
            alpha=args.rmu_alpha,
            local_files_only=args.local_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

    summary = {
        "algo": algo,
        "model": args.model,
        "run_dir": run_dir,
        "adapter_dir": adapter_dir,
        "pokemon_bench": bench_path,
        "forget_traits": FORGET_TRAITS,
        "retain_traits": RETAIN_TRAITS,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "beta": args.beta if algo == "dpo" else None,
        "layer_index": args.layer_index if algo == "rmu" else None,
        "rmu_c": args.rmu_c if algo == "rmu" else None,
        "rmu_alpha": args.rmu_alpha if algo == "rmu" else None,
        "local_model": bool(args.local_model),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    summary_path = os.path.join(run_dir, "part2_run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[part2] Run summary saved to {summary_path}")
    print(
        "[part2] To evaluate this unlearned model with Part 1, "
        "call part_1.py with --model pointing to the adapter directory "
        "and --local_model."
    )
    print(f"[part2] Example model path: {adapter_dir}")

def _load_part1_results(part1_dir: str) -> List[dict]:
    files = []
    for fname in os.listdir(part1_dir):
        if fname.startswith("model_") and fname.endswith(".json"):
            files.append(os.path.join(part1_dir, fname))

    if not files:
        raise FileNotFoundError(
            f"No model_*.json files found in {part1_dir}; "
            f"run part_1.py evaluations first."
        )

    records: List[dict] = []
    for path in files:
        with open(path) as f:
            rec = json.load(f)
            records.append(rec)
    return records

def run_analysis(args):
    ensure_dir(args.outdir)

    print(f"[part2] Loading Part 1 results from {args.part1_dir} ...")
    all_results = _load_part1_results(args.part1_dir)

    if args.analysis_models:
        selected = []
        by_model = {r["model"]: r for r in all_results}
        for m in args.analysis_models:
            if m not in by_model:
                print(
                    f"[part2] Warning: requested model '{m}' not found "
                    f"in Part 1 results."
                )
                continue
            selected.append(by_model[m])
        all_results = selected

    if not all_results:
        print("[part2] No results left after filtering; nothing to analyze.")
        return

    all_results.sort(key=lambda x: x.get("size_b", 0.0))

    if args.analysis_labels:
        if len(args.analysis_labels) != len(all_results):
            raise ValueError(
                "Number of --analysis_labels must match number "
                "of selected models (after any filtering)."
            )
        labels = list(args.analysis_labels)
    else:
        labels = [r["model"] for r in all_results]

    summary_rows = []
    for label, rec in zip(labels, all_results):
        row = {
            "label": label,
            "model": rec["model"],
            "size_b": rec.get("size_b", 0.0),
            "Type1": rec.get("Type1", 0.0),
            "HP": rec.get("HP", 0.0),
            "Speed": rec.get("Speed", 0.0),
            "Defense": rec.get("Defense", 0.0),
            "IFEval_inst_strict": rec.get("IFEval_inst_strict", 0.0),
        }
        summary_rows.append(row)

    csv_path = os.path.join(args.outdir, "part2_unlearning_summary.csv")
    json_path = os.path.join(args.outdir, "part2_unlearning_summary.json")
    save_table(summary_rows, csv_path, json_path)

    print("\nIFEval instruction-level strict accuracy (used for Part 2 tables/plots):")
    label_width = max(len("Label"), max(len(r["label"]) for r in summary_rows)) + 2
    model_width = max(len("Model"), max(len(r["model"]) for r in summary_rows)) + 2
    header = (
        f"{'Label':<{label_width}} {'Model':<{model_width}} "
        f"{'InstStrict':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['label']:<{label_width}} "
            f"{row['model']:<{model_width}} "
            f"{row['IFEval_inst_strict']:>12.4f}"
        )

    print("\nFull IFEval metrics per model (printed only, not used for Part 2 plots):")
    header_full = (
        f"{'Label':<{label_width}} "
        f"{'PromptStrict':>14} {'InstStrict':>12} "
        f"{'PromptLoose':>14} {'InstLoose':>12}"
    )
    print(header_full)
    print("-" * len(header_full))
    for label, rec in zip(labels, all_results):
        ps = rec.get("IFEval_prompt_strict", 0.0)
        is_ = rec.get("IFEval_inst_strict", 0.0)
        pl = rec.get("IFEval_prompt_loose", 0.0)
        il = rec.get("IFEval_inst_loose", 0.0)
        print(
            f"{label:<{label_width}} "
            f"{ps:>14.4f} {is_:>12.4f} {pl:>14.4f} {il:>12.4f}"
        )

    inst_values = [r["IFEval_inst_strict"] for r in summary_rows]
    plot_path = os.path.join(args.outdir, "part2_ifeval_inst_strict.png")
    plot_ifeval_inst_strict_by_model(labels, inst_values, plot_path)

    print(
        f"\n[part2] Part 2 summary saved to:\n"
        f"  - {csv_path}\n"
        f"  - {json_path}\n"
        f"  - {plot_path}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results/part_2")
    ap.add_argument(
        "--analysis",
        action="store_true",
        help="Run analysis using results from part_1 (no training).",
    )

    # Shared
    ap.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="Base model ID or local path for unlearning.",
    )
    ap.add_argument(
        "--local_model",
        action="store_true",
        help="Treat --model as a local directory for loading weights/tokenizer.",
    )
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    # Pokemon benchmark configuration
    ap.add_argument("--pokemon_csv", type=str, default="data/pokemon.csv")
    ap.add_argument(
        "--pokemon_bench",
        type=str,
        default="data/pokemon_benchmark_mcq.csv",
        help="MCQ benchmark CSV used for forget/retain sets.",
    )
    ap.add_argument("--min_per_trait", type=int, default=500)

    # DPO/NPO-specific
    ap.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="NPO temperature parameter for DPO-style unlearning.",
    )

    # RMU-specific
    ap.add_argument(
        "--layer_index",
        type=int,
        default=4,
        help="Hidden layer index used in RMU representation loss.",
    )
    ap.add_argument(
        "--rmu_c",
        type=float,
        default=6.5,
        help="Scaling factor c in the RMU forget loss.",
    )
    ap.add_argument(
        "--rmu_alpha",
        type=float,
        default=1200.0,
        help="Weight on the RMU retain loss.",
    )

    # Analysis settings
    ap.add_argument(
        "--part1_dir",
        type=str,
        default="results/part1",
        help="Directory holding part_1 model_*.json result files.",
    )
    ap.add_argument(
        "--analysis_models",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of model identifiers (as stored in part_1 results) "
            "to include in Part 2 analysis. If omitted, all models are used."
        ),
    )
    ap.add_argument(
        "--analysis_labels",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of labels for the selected models, used in tables "
            "and plots. Must match the number of selected models."
        ),
    )

    ap.add_argument(
        "--algo",
        type=str,
        default="dpo",
        choices=["dpo", "rmu"],
        help="Unlearning algorithm to run (training mode).",
    )

    args = ap.parse_args()

    if args.analysis:
        run_analysis(args)
    else:
        run_training(args)

if __name__ == "__main__":
    main()

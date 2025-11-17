import argparse
import json
import os

from data.pokemon import build_pokemon_benchmark
from unlearning.ga import run_ga_unlearning
from unlearning.gd import run_gd_unlearning
from utils.part_1 import ensure_dir, sanitize_filename, save_table, get_model_label
from utils.part_2 import plot_ifeval_inst_strict_by_model

FORGET_TRAITS = ["Type 1", "HP", "Defense"]

def run_training(args):
    ensure_dir(args.outdir)

    bench_path = args.pokemon_bench
    if not os.path.exists(bench_path):
        print(
            "[part2] Pokemon MCQ benchmark not found at {}, building it...".format(
                bench_path
            )
        )
        ensure_dir(os.path.dirname(bench_path) or ".")
        build_pokemon_benchmark(
            raw_csv_path=args.pokemon_csv,
            out_csv_path=bench_path,
            use_mcq=True,
            min_per_trait=args.min_per_trait,
        )
        print("[part2] Pokemon MCQ benchmark created.")

    algo = args.algo.lower()
    if algo not in {"ga", "gd"}:
        raise ValueError("Unknown algo: {}".format(algo))

    model_tag = sanitize_filename(args.model)
    run_tag = "{}_pokemon_{}".format(algo, model_tag)
    run_dir = os.path.join(args.outdir, run_tag)
    ensure_dir(run_dir)

    print("[part2] Running {} unlearning for model={}".format(algo.upper(), args.model))
    print("[part2] Forget traits: {}".format(FORGET_TRAITS))
    print("[part2] Output directory: {}".format(run_dir))

    if algo == "ga":
        model_dir = run_ga_unlearning(
            model_id=args.model,
            pokemon_bench_path=bench_path,
            outdir=run_dir,
            forget_traits=FORGET_TRAITS,
            lr=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            local_files_only=args.local_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    else:
        model_dir = run_gd_unlearning(
            model_id=args.model,
            pokemon_bench_path=bench_path,
            outdir=run_dir,
            forget_traits=FORGET_TRAITS,
            retain_dataset_name=args.retain_dataset_name,
            retain_split=args.retain_split,
            retain_max_examples=args.retain_max_examples,
            lr=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lambda_retain=args.lambda_retain,
            local_files_only=args.local_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

    summary = {
        "algo": algo,
        "model": args.model,
        "run_dir": run_dir,
        "model_dir": model_dir,
        "pokemon_bench": bench_path,
        "forget_traits": FORGET_TRAITS,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "local_model": bool(args.local_model),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    if algo == "gd":
        summary["retain_dataset_name"] = args.retain_dataset_name
        summary["retain_split"] = args.retain_split
        summary["retain_max_examples"] = args.retain_max_examples
        summary["lambda_retain"] = args.lambda_retain

    summary_path = os.path.join(run_dir, "part2_run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("[part2] Run summary saved to {}".format(summary_path))
    print(
        "[part2] To evaluate this unlearned model with Part 1, "
        "call part_1.py with --model pointing to the model directory "
        "and --local_model."
    )
    print("[part2] Example model path: {}".format(model_dir))

def _load_part1_results(part1_dir):
    files = []
    for fname in os.listdir(part1_dir):
        if fname.startswith("model_") and fname.endswith(".json"):
            files.append(os.path.join(part1_dir, fname))

    if not files:
        raise FileNotFoundError(
            "No model_*.json files found in {}; run part_1.py evaluations first.".format(
                part1_dir
            )
        )

    records = []
    for path in files:
        with open(path) as f:
            rec = json.load(f)
            records.append(rec)
    return records

def run_analysis(args):
    ensure_dir(args.outdir)

    print("[part2] Loading Part 1 results from {} ...".format(args.part1_dir))
    all_results = _load_part1_results(args.part1_dir)

    if args.analysis_models:
        selected = []
        by_model = {r["model"]: r for r in all_results}
        for m in args.analysis_models:
            if m not in by_model:
                print(
                    "[part2] Warning: requested model '{}' not found in Part 1 results.".format(
                        m
                    )
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
                "Number of --analysis_labels must match number of selected models."
            )
        labels = list(args.analysis_labels)
    else:
        labels = [get_model_label(r["model"]) for r in all_results]

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
        "{:<{lw}} {:<{mw}} {:>12}".format(
            "Label", "Model", "InstStrict", lw=label_width, mw=model_width
        )
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            "{:<{lw}} {:<{mw}} {:>12.4f}".format(
                row["label"],
                row["model"],
                row["IFEval_inst_strict"],
                lw=label_width,
                mw=model_width,
            )
        )

    print(
        "\nFull IFEval metrics per model (printed only, not used for Part 2 plots):"
    )
    header_full = (
        "{:<{lw}} {:>14} {:>12} {:>14} {:>12}".format(
            "Label", "PromptStrict", "InstStrict", "PromptLoose", "InstLoose", lw=label_width
        )
    )
    print(header_full)
    print("-" * len(header_full))
    for label, rec in zip(labels, all_results):
        ps = rec.get("IFEval_prompt_strict", 0.0)
        is_ = rec.get("IFEval_inst_strict", 0.0)
        pl = rec.get("IFEval_prompt_loose", 0.0)
        il = rec.get("IFEval_inst_loose", 0.0)
        print(
            "{:<{lw}} {:>14.4f} {:>12.4f} {:>14.4f} {:>12.4f}".format(
                label, ps, is_, pl, il, lw=label_width
            )
        )

    inst_values = [r["IFEval_inst_strict"] for r in summary_rows]
    plot_path = os.path.join(args.outdir, "part2_ifeval_inst_strict.png")
    plot_ifeval_inst_strict_by_model(labels, inst_values, plot_path)

    print(
        "\n[part2] Part 2 summary saved to:\n"
        "  - {}\n"
        "  - {}\n"
        "  - {}".format(csv_path, json_path, plot_path)
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results/part_2")
    ap.add_argument(
        "--analysis",
        action="store_true",
        help="Run analysis using results from part_1 (no training).",
    )

    # Shared training args
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
    ap.add_argument("--num_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    # Pokemon benchmark configuration
    ap.add_argument("--pokemon_csv", type=str, default="data/pokemon.csv")
    ap.add_argument(
        "--pokemon_bench",
        type=str,
        default="data/pokemon_benchmark_mcq.csv",
        help="MCQ benchmark CSV used for forget set.",
    )
    ap.add_argument("--min_per_trait", type=int, default=500)

    # GD-specific retain set configuration
    ap.add_argument(
        "--retain_dataset_name",
        type=str,
        default="tatsu-lab/alpaca",
        help="HuggingFace dataset name for GD retain set.",
    )
    ap.add_argument(
        "--retain_split",
        type=str,
        default="train",
        help="Dataset split for GD retain set.",
    )
    ap.add_argument(
        "--retain_max_examples",
        type=int,
        default=5000,
        help="Maximum number of retain examples to use (0 = all).",
    )
    ap.add_argument(
        "--lambda_retain",
        type=float,
        default=1.0,
        help="Weight on retain loss term in GD.",
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
        default="ga",
        choices=["ga", "gd"],
        help="Unlearning algorithm to run (training mode).",
    )

    args = ap.parse_args()

    if args.analysis:
        run_analysis(args)
    else:
        run_training(args)

if __name__ == "__main__":
    main()

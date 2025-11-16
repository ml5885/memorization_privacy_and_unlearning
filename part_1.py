import argparse
import json
import os

from data.pokemon import (
    build_pokemon_benchmark,
    TRAIT_PLOT_LABELS,
    run_pokemon_eval,
)
from data.triviaqa import run_triviaqa_eval
from data.ifeval import run_ifeval_eval
from models.gemma import load_gemma_model
from utils.part_1 import (
    ensure_dir,
    sanitize_filename,
    plot_memorization,
    save_table,
)


def print_table(all_results):
    model_col_width = max(len("Model"), max(len(r["model"]) for r in all_results)) + 2
    size_col_width = 8
    metric_col_width = 10

    print("\nAccuracy Metrics:")
    header = f"{'Model':<{model_col_width}} {'Size(B)':<{size_col_width}}"
    for label in TRAIT_PLOT_LABELS.values():
        header += f" {label:<{metric_col_width}}"
    header += f" {'TriviaQA':<{metric_col_width}}"
    print(header)
    print("-" * len(header))

    for result in all_results:
        row = f"{result['model']:<{model_col_width}} {result['size_b']:<{size_col_width}.1f}"
        for label in TRAIT_PLOT_LABELS.values():
            acc = result.get(label, 0.0)
            row += f" {acc:<{metric_col_width}.4f}"
        row += f" {result.get('TriviaQA', 0.0):<{metric_col_width}.4f}"
        print(row)

    print("\nIFEval Metrics:")
    ifeval_col_width = 15
    ifeval_header = f"{'Model':<{model_col_width}} {'Size(B)':<{size_col_width}}"
    ifeval_header += f" {'Prompt Strict':<{ifeval_col_width}} {'Inst Strict':<{ifeval_col_width}}"
    ifeval_header += f" {'Prompt Loose':<{ifeval_col_width}} {'Inst Loose':<{ifeval_col_width}}"
    print(ifeval_header)
    print("-" * len(ifeval_header))

    for result in all_results:
        row = f"{result['model']:<{model_col_width}} {result['size_b']:<{size_col_width}.1f}"
        row += f" {result.get('IFEval_prompt_strict', 0.0):<{ifeval_col_width}.4f}"
        row += f" {result.get('IFEval_inst_strict', 0.0):<{ifeval_col_width}.4f}"
        row += f" {result.get('IFEval_prompt_loose', 0.0):<{ifeval_col_width}.4f}"
        row += f" {result.get('IFEval_inst_loose', 0.0):<{ifeval_col_width}.4f}"
        print(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results/part1")
    ap.add_argument(
        "--analysis",
        action="store_true",
        help="Run analysis only (plot results from existing model outputs)",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID or local path to evaluate (e.g., google/gemma-3-1b-it)",
    )
    ap.add_argument(
        "--local_model",
        action="store_true",
        help="Treat --model as a local model directory (no HuggingFace download)",
    )
    ap.add_argument(
        "--model_size",
        type=float,
        default=None,
        help="Model size in billions of parameters (e.g., 1, 4, 12)",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "pokemon", "triviaqa", "ifeval"],
        help="Which dataset(s) to evaluate on (default: all)",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit_pokemon", type=int, default=0)
    ap.add_argument("--limit_triviaqa", type=int, default=5000)
    ap.add_argument("--limit_ifeval", type=int, default=0)
    ap.add_argument("--pokemon_csv", type=str, default="data/pokemon.csv")
    ap.add_argument("--pokemon_mcq", action="store_true")
    ap.add_argument("--min_per_trait", type=int, default=500)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    if args.pokemon_mcq:
        bench_path = os.path.join("data", "pokemon_benchmark_mcq.csv")
    else:
        bench_path = os.path.join("data", "pokemon_benchmark.csv")

    if args.analysis:
        print(f"\n{'#'*80}")
        print("Running analysis mode: generating plots from existing results")
        print(f"{'#'*80}\n")

        results_files = []
        for fname in os.listdir(args.outdir):
            if fname.startswith("model_") and fname.endswith(".json"):
                results_files.append(os.path.join(args.outdir, fname))

        if not results_files:
            print(f"No model result files found in {args.outdir}")
            print("Looking for files matching pattern: model_*.json")
            return

        print(f"Found {len(results_files)} model result file(s):")
        for f in results_files:
            print(f"  - {os.path.basename(f)}")

        all_results = []
        for result_file in results_files:
            with open(result_file) as f:
                result = json.load(f)
                all_results.append(result)

        all_results.sort(key=lambda x: x.get("size_b", 0))

        save_table(
            all_results,
            os.path.join(args.outdir, "part1_results.csv"),
            os.path.join(args.outdir, "part1_results.json"),
        )
        print(
            f"\nCombined results saved to {args.outdir}/part1_results.csv and part1_results.json"
        )

        print(f"\n{'='*80}")
        print("Results Summary")
        print(f"{'='*80}")

        print_table(all_results)

        print(f"\n{'='*80}\n")

        print("Generating memorization plot...")

        sizes = [r["size_b"] for r in all_results]
        trait2acc = {
            label: [r.get(label, 0.0) for r in all_results]
            for label in TRAIT_PLOT_LABELS.values()
        }

        plot_path = os.path.join(args.outdir, "part1_memorization.png")
        plot_memorization(
            sizes,
            trait2acc,
            plot_path,
        )
        print(f"Plot saved to {plot_path}")

    else:
        if not args.model:
            print("Error: --model argument is required for evaluation mode")
            print(
                "Example: python part_1.py --model google/gemma-3-1b-it --model_size 1"
            )
            return

        if args.model_size is None:
            print("Error: --model_size argument is required for evaluation mode")
            print(
                "Example: python part_1.py --model google/gemma-3-1b-it --model_size 1"
            )
            return

        model_id = args.model
        model_size = args.model_size

        print(f"\n{'#'*80}")
        print(f"Running evaluation for model: {model_id} ({model_size}B parameters)")
        print(f"{'#'*80}\n")

        if not os.path.exists(bench_path):
            print(f"Building Pokemon benchmark at {bench_path}...")
            ensure_dir(os.path.dirname(bench_path) or ".")
            build_pokemon_benchmark(
                raw_csv_path=args.pokemon_csv,
                out_csv_path=bench_path,
                use_mcq=args.pokemon_mcq,
                min_per_trait=args.min_per_trait,
            )
            print("Pokemon benchmark created successfully!")

        responses_dir = os.path.join(args.outdir, "responses")
        ensure_dir(responses_dir)

        print(f"\nLoading model: {model_id}...")
        tokenizer, model = load_gemma_model(
            model_id,
            local_files_only=args.local_model,
        )
        print("Model loaded successfully!")

        mid_safe = sanitize_filename(model_id)

        trait_acc = {}
        tqa = 0.0
        ife_scores = {
            "prompt_strict": 0.0,
            "inst_strict": 0.0,
            "prompt_loose": 0.0,
            "inst_loose": 0.0,
        }

        if args.dataset in ["all", "pokemon"]:
            trait_acc = run_pokemon_eval(
                tokenizer,
                model,
                model_id,
                bench_path,
                args.batch_size,
                args.limit_pokemon,
                save_path=os.path.join(
                    responses_dir,
                    f"pokemon_{mid_safe}.jsonl",
                ),
                eval_mode="auto",
            )

        if args.dataset in ["all", "triviaqa"]:
            tqa = run_triviaqa_eval(
                tokenizer,
                model,
                model_id,
                batch_size=args.batch_size,
                limit=args.limit_triviaqa,
                save_path=os.path.join(
                    responses_dir,
                    f"triviaqa_rc_{mid_safe}.jsonl",
                ),
                eval_mode="contains",
            )

        if args.dataset in ["all", "ifeval"]:
            ife_scores = run_ifeval_eval(
                tokenizer,
                model,
                model_id,
                batch_size=args.batch_size,
                limit=args.limit_ifeval,
                save_path=os.path.join(
                    responses_dir,
                    f"ifeval_{mid_safe}.jsonl",
                ),
            )

        result = {"model": model_id, "size_b": model_size}
        for trait_key, label in TRAIT_PLOT_LABELS.items():
            result[label] = round(trait_acc.get(trait_key, 0.0), 4)
        result["TriviaQA"] = round(tqa, 4)

        result["IFEval"] = round(ife_scores["prompt_strict"], 4)
        result["IFEval_prompt_strict"] = round(ife_scores["prompt_strict"], 4)
        result["IFEval_inst_strict"] = round(ife_scores["inst_strict"], 4)
        result["IFEval_prompt_loose"] = round(ife_scores["prompt_loose"], 4)
        result["IFEval_inst_loose"] = round(ife_scores["inst_loose"], 4)

        model_result_path = os.path.join(args.outdir, f"model_{mid_safe}.json")
        with open(model_result_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n{'#'*80}")
        print(f"Evaluation completed for {model_id}")
        print(f"Dataset(s) evaluated: {args.dataset}")
        print(f"{'#'*80}")
        summary_parts = []
        if args.dataset in ["all", "triviaqa"]:
            summary_parts.append(f"TriviaQA={tqa:.4f}")
        if args.dataset in ["all", "ifeval"]:
            summary_parts.append(
                f"IFEval[prompt_strict]={ife_scores['prompt_strict']:.4f}, "
                f"IFEval[inst_strict]={ife_scores['inst_strict']:.4f}, "
                f"IFEval[prompt_loose]={ife_scores['prompt_loose']:.4f}, "
                f"IFEval[inst_loose]={ife_scores['inst_loose']:.4f}"
            )
        if summary_parts:
            print("Summary: " + ", ".join(summary_parts))

if __name__ == "__main__":
    main()

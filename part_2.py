import argparse
import json
import os
IS_MAIN = os.environ.get("RANK", "0") == "0" or os.environ.get("LOCAL_RANK", "0") == "0"

import random

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from tqdm import tqdm

from data.pokemon import (
    build_pokemon_benchmark,
    load_pokemon_benchmark,
    TRAIT_PLOT_LABELS,
    run_pokemon_eval,
)
from data.triviaqa import run_triviaqa_eval
from data.ifeval import run_ifeval_eval
from models.gemma import load_gemma_model, generate_batch
from utils.part_1 import ensure_dir, sanitize_filename, open_jsonl, write_jsonl, save_table
from utils.part_2 import (
    default_refusal,
    split_forget_holdout,
    make_forget_pairs,
    pack_result_row,
    plot_grouped,
)

from unlearning.dpo import train_dpo_unlearning
from unlearning.rmu import compute_pca_directions, train_rmu_unlearning

def _load_or_build_benchmark(args):
    if args.pokemon_mcq:
        bench_path = os.path.join("data", "pokemon_benchmark_mcq.csv")
    else:
        bench_path = os.path.join("data", "pokemon_benchmark.csv")
    
    if not os.path.exists(bench_path):
        ensure_dir(os.path.dirname(bench_path) or ".")
        build_pokemon_benchmark(
            raw_csv_path=args.pokemon_csv,
            out_csv_path=bench_path,
            use_mcq=args.pokemon_mcq,
            min_per_trait=args.min_per_trait,
        )
    
    return bench_path

def _sample_text_prompts_for_pca(tokenizer, model, n, batch_size):
    ds = load_dataset("google/IFEval", split="train")
    prompts = []
    
    for it in ds:
        p = str(it.get("prompt") or "").strip()
        if p:
            prompts.append(p)
        if len(prompts) >= n:
            break
    
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i+batch_size]
        outs.extend(generate_batch(tokenizer, model, chunk))
    
    return prompts

def _avg_forget_accuracy(trait_acc, held_out_trait):
    ks = [k for k in trait_acc if k != held_out_trait]
    if not ks:
        return 0.0
    return sum(trait_acc[k] for k in ks) / len(ks)

def evaluate_all(tokenizer, model, model_id, bench_path, outdir, limits, save_prefix):
    responses_dir = os.path.join(outdir, "responses")
    ensure_dir(responses_dir)
    
    trait_acc = run_pokemon_eval(
        tokenizer, model, model_id, bench_path,
        batch_size=limits["batch_size"],
        limit=limits["limit_pokemon"],
        save_path=os.path.join(responses_dir, f"{save_prefix}_pokemon.jsonl"),
        eval_mode="auto",
    )
    
    tqa = run_triviaqa_eval(
        tokenizer, model, model_id,
        batch_size=limits["batch_size"],
        limit=limits["limit_triviaqa"],
        save_path=os.path.join(responses_dir, f"{save_prefix}_triviaqa.jsonl"),
        eval_mode="contains",
    )
    
    ife = run_ifeval_eval(
        tokenizer, model, model_id,
        batch_size=limits["batch_size"],
        limit=limits["limit_ifeval"],
        save_path=os.path.join(responses_dir, f"{save_prefix}_ifeval.jsonl"),
    )
    
    return trait_acc, tqa, ife

import os
from accelerate import Accelerator

IS_MAIN = os.environ.get("RANK", "0") == "0"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results/part2")
    ap.add_argument("--analysis", action="store_true")
    ap.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    ap.add_argument("--model_size", type=float, default=4.0)
    ap.add_argument("--held_out_trait", type=str, default="Speed")
    ap.add_argument("--pokemon_csv", type=str, default="data/pokemon.csv")
    ap.add_argument("--pokemon_mcq", action="store_true")
    ap.add_argument("--min_per_trait", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit_pokemon", type=int, default=0)
    ap.add_argument("--limit_triviaqa", type=int, default=0)
    ap.add_argument("--limit_ifeval", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rmu_k", type=int, default=16)
    ap.add_argument("--rmu_layer", type=int, default=-2)
    ap.add_argument("--rmu_alpha", type=float, default=1.0)
    ap.add_argument("--pca_examples", type=int, default=1024)
    ap.add_argument("--method", type=str, default="both", choices=["both", "dpo", "rmu"])
    args = ap.parse_args()

    accel = Accelerator()
    ensure_dir(args.outdir)

    # Analysis-only: rank 0 aggregates/plots; others idle
    if args.analysis:
        if IS_MAIN:
            rows = []
            for fn in os.listdir(args.outdir):
                if fn.endswith(".json") and fn.startswith("part2_"):
                    with open(os.path.join(args.outdir, fn)) as f:
                        obj = json.load(f)
                    if isinstance(obj, dict) and "model" in obj:
                        rows.append(obj)
            if not rows:
                print("No Part 2 result files found.")
                return
            rows.sort(key=lambda r: r["model"])
            save_table(
                rows,
                os.path.join(args.outdir, "part2_results.csv"),
                os.path.join(args.outdir, "part2_results.json"),
            )
            grouped = {}
            for r in rows:
                forget = (r.get("Type1", 0.0) + r.get("HP", 0.0) + r.get("Defense", 0.0)) / 3.0
                heldout = r.get("Speed", 0.0)
                general = (r.get("TriviaQA", 0.0) + r.get("IFEval_inst_strict", 0.0)) / 2.0
                grouped[r["model"]] = {"forget": forget, "heldout": heldout, "general": general}
            plot_grouped(grouped, os.path.join(args.outdir, "part2_grouped.png"))
        accel.wait_for_everyone()
        return

    # Benchmark path + build only on rank 0, then sync
    if args.pokemon_mcq:
        bench_path = os.path.join("data", "pokemon_benchmark_mcq.csv")
    else:
        bench_path = os.path.join("data", "pokemon_benchmark.csv")
    if IS_MAIN and not os.path.exists(bench_path):
        ensure_dir(os.path.dirname(bench_path) or ".")
        build_pokemon_benchmark(
            raw_csv_path=args.pokemon_csv,
            out_csv_path=bench_path,
            use_mcq=args.pokemon_mcq,
            min_per_trait=args.min_per_trait,
        )
    accel.wait_for_everyone()

    print(f"\n{'#'*80}\nUnlearning on: {args.model} | Held-out: {args.held_out_trait}\n{'#'*80}\n")
    tokenizer, base_model = load_gemma_model(args.model)
    mid_safe = sanitize_filename(args.model)

    all_items = load_pokemon_benchmark(bench_path)
    forget_raw, holdout_raw = split_forget_holdout(all_items, args.held_out_trait)
    forget_pairs = make_forget_pairs(forget_raw)

    limits = {
        "batch_size": args.batch_size,
        "limit_pokemon": args.limit_pokemon,
        "limit_triviaqa": args.limit_triviaqa,
        "limit_ifeval": args.limit_ifeval,
    }

    rows = []

    # Baseline eval only on rank 0
    if IS_MAIN:
        print("\n[base] Evaluating baseline...")
        trait_acc_b, tqa_b, ife_b = evaluate_all(
            tokenizer, base_model, f"{args.model}", bench_path, args.outdir, limits, save_prefix=f"base_{mid_safe}"
        )
        base_row = pack_result_row("Base", args.model_size, trait_acc_b, tqa_b, ife_b)
        with open(os.path.join(args.outdir, f"part2_Base.json"), "w") as f:
            json.dump(base_row, f, indent=2)
        rows.append(base_row)
    accel.wait_for_everyone()

    # DPO
    if args.method in {"both", "dpo"}:
        print("\n[DPO] Training...")
        ref_model = load_gemma_model(args.model)[1]
        dpo_ckpt = os.path.join(args.outdir, "checkpoints", "dpo")
        ensure_dir(dpo_ckpt)
        dpo_model = train_dpo_unlearning(
            tokenizer=tokenizer,
            model=base_model,
            ref_model=ref_model,
            pairs=forget_pairs,
            epochs=args.epochs,
            lr=args.lr,
            beta=args.beta,
            batch_size=args.batch_size,
            save_dir=dpo_ckpt,
        )
        accel.wait_for_everyone()
        if IS_MAIN:
            print("\n[DPO] Evaluating...")
            trait_acc_d, tqa_d, ife_d = evaluate_all(
                tokenizer, dpo_model, f"{args.model}-dpo", bench_path, args.outdir, limits, save_prefix=f"dpo_{mid_safe}"
            )
            dpo_row = pack_result_row("DPO Unlearned", args.model_size, trait_acc_d, tqa_d, ife_d)
            with open(os.path.join(args.outdir, f"part2_DPO.json"), "w") as f:
                json.dump(dpo_row, f, indent=2)
            rows.append(dpo_row)
        accel.wait_for_everyone()

    # RMU
    if args.method in {"both", "rmu"}:
        print("\n[RMU] Computing PCA directions...")
        pca_prompts = _sample_text_prompts_for_pca(tokenizer, base_model, args.pca_examples, args.batch_size)
        U = compute_pca_directions(tokenizer, base_model, pca_prompts, layer=args.rmu_layer, k=args.rmu_k, batch_size=args.batch_size)
        rmu_ckpt = os.path.join(args.outdir, "checkpoints", "rmu")
        ensure_dir(rmu_ckpt)
        print("\n[RMU] Training...")
        rmu_model = train_rmu_unlearning(
            tokenizer=tokenizer,
            model=base_model,
            forget_records=forget_raw,
            refusal=default_refusal(),
            U=U,
            alpha=args.rmu_alpha,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            layer=args.rmu_layer,
            save_dir=rmu_ckpt,
        )
        accel.wait_for_everyone()
        if IS_MAIN:
            print("\n[RMU] Evaluating...")
            trait_acc_r, tqa_r, ife_r = evaluate_all(
                tokenizer, rmu_model, f"{args.model}-rmu", bench_path, args.outdir, limits, save_prefix=f"rmu_{mid_safe}"
            )
            rmu_row = pack_result_row("RMU Unlearned", args.model_size, trait_acc_r, tqa_r, ife_r)
            with open(os.path.join(args.outdir, f"part2_RMU.json"), "w") as f:
                json.dump(rmu_row, f, indent=2)
            rows.append(rmu_row)
        accel.wait_for_everyone()

    if IS_MAIN and rows:
        save_table(
            rows,
            os.path.join(args.outdir, "part2_results.csv"),
            os.path.join(args.outdir, "part2_results.json"),
        )
        grouped = {}
        for r in rows:
            forget = (r.get("Type1", 0.0) + r.get("HP", 0.0) + r.get("Defense", 0.0)) / 3.0
            heldout = r.get("Speed", 0.0)
            general = (r.get("TriviaQA", 0.0) + r.get("IFEval_inst_strict", 0.0)) / 2.0
            grouped[r["model"]] = {"forget": forget, "heldout": heldout, "general": general}
        plot_grouped(grouped, os.path.join(args.outdir, "part2_grouped.png"))

if __name__ == "__main__":
    main()

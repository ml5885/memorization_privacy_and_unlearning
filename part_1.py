import argparse
import json
import os
import re
import string
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

from data.pokemon import (
    build_pokemon_benchmark,
    load_pokemon_benchmark,
    TRAIT_COLUMN_MAP,
    TRAIT_PLOT_LABELS,
)
from models.gemma import load_gemma_model, generate_batch
from utils.part_1 import (
    ensure_dir,
    sanitize_filename,
    plot_memorization,
    save_table,
    open_jsonl,
    write_jsonl,
)

def _strip_punct(text: str) -> str:
    table = str.maketrans({c: " " for c in string.punctuation})
    return text.translate(table)

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _strip_punct(s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def eval_exact_match(pred, golds):
    p = normalize_text(pred)
    for g in golds:
        if p == normalize_text(str(g)):
            return 1
    return 0

def eval_contains(pred, golds):
    p = normalize_text(pred)
    for g in golds:
        gnorm = normalize_text(g)
        if not gnorm:
            continue
        if gnorm in p:
            return 1
    return 0

def eval_prefix(pred, golds):
    p = normalize_text(pred)
    for g in golds:
        gnorm = normalize_text(g)
        if not gnorm:
            continue
        if p.startswith(gnorm):
            return 1
    return 0

def _first_number(text: str):
    m = re.search(r"[-+]?(?:\d+\.\d+|\d+)", text or "")
    return m.group(0) if m else None

def eval_numeric(pred, golds):
    pnum = _first_number(pred)
    if pnum is None:
        return 0
    for g in golds:
        gnum = _first_number(str(g))
        if gnum is not None and str(gnum) == str(pnum):
            return 1
    return 0

def evaluate_answer(pred, golds, mode: str):
    mode = (mode or "auto").lower()
    if mode == "exact":
        return eval_exact_match(pred, golds)
    if mode == "contains":
        return eval_contains(pred, golds)
    if mode == "prefix":
        return eval_prefix(pred, golds)
    if mode == "numeric":
        return eval_numeric(pred, golds)
    for g in golds:
        if _first_number(str(g)) is not None and normalize_text(str(g)).strip() == _first_number(str(g)):
            return eval_numeric(pred, golds)
    return eval_contains(pred, golds)

def eval_mcq_letter(pred, correct_letter):
    p = pred.strip().upper()
    if len(p) > 0:
        p = p[0]
    return 1 if p == correct_letter.upper() else 0

def run_pokemon_eval(tokenizer, model, model_id, bench_path, batch_size, limit, save_path=None, eval_mode: str = "auto"):
    examples = load_pokemon_benchmark(bench_path)
    if limit > 0:
        examples = examples[:limit]

    trait2correct = defaultdict(int)
    trait2total = defaultdict(int)

    log_f = open_jsonl(save_path) if save_path else None

    batch_prompts, batch_meta = [], []
    for ex in tqdm(examples, desc=f"{model_id} / pokemon"):
        batch_prompts.append(ex["prompt"])
        batch_meta.append(ex)

        if len(batch_prompts) == batch_size:
            outs = generate_batch(tokenizer, model, batch_prompts)
            for out, meta in zip(outs, batch_meta):
                if meta["format"] == "mcq":
                    trait2correct[meta["trait"]] += eval_mcq_letter(
                        out, meta["answer_letter"]
                    )
                else:
                    trait2correct[meta["trait"]] += evaluate_answer(
                        out, [meta["answer"]], eval_mode
                    )
                trait2total[meta["trait"]] += 1

                if log_f:
                    rec = {
                        "dataset": "pokemon",
                        "model": model_id,
                        "trait": meta.get("trait"),
                        "name": meta.get("name"),
                        "format": meta.get("format"),
                        "prompt": meta.get("prompt"),
                        "gold_answer": meta.get("answer"),
                        "gold_letter": meta.get("answer_letter"),
                        "prediction": out,
                    }
                    write_jsonl(log_f, rec)

            batch_prompts, batch_meta = [], []

    if batch_prompts:
        outs = generate_batch(tokenizer, model, batch_prompts)
        for out, meta in zip(outs, batch_meta):
            if meta["format"] == "mcq":
                trait2correct[meta["trait"]] += eval_mcq_letter(
                    out, meta["answer_letter"]
                )
            else:
                trait2correct[meta["trait"]] += evaluate_answer(out, [meta["answer"]], eval_mode)
            trait2total[meta["trait"]] += 1

            if log_f:
                rec = {
                    "dataset": "pokemon",
                    "model": model_id,
                    "trait": meta.get("trait"),
                    "name": meta.get("name"),
                    "format": meta.get("format"),
                    "prompt": meta.get("prompt"),
                    "gold_answer": meta.get("answer"),
                    "gold_letter": meta.get("answer_letter"),
                    "prediction": out,
                }
                write_jsonl(log_f, rec)

    if log_f:
        log_f.close()

    return {
        t: trait2correct[t] / max(1, trait2total[t])
        for t in sorted(trait2total.keys())
    }

def run_triviaqa_eval(tokenizer, model, model_id, batch_size, limit, save_path=None, eval_mode: str = "contains"):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")

    records = []
    for item in ds:
        q = item["question"]
        golds = [item["answer"]["value"]] + item["answer"].get("aliases", [])
        prompt = f"Question: {q}\nAnswer with just the final answer (no punctuation or explanation):"
        records.append({"prompt": prompt, "golds": golds, "question": q})
        if limit > 0 and len(records) >= limit:
            break

    correct, total = 0, 0
    log_f = open_jsonl(save_path) if save_path else None
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        prompts = [r["prompt"] for r in chunk]
        outs = generate_batch(tokenizer, model, prompts)
        for out, rec in zip(outs, chunk):
            score = evaluate_answer(out, rec["golds"], eval_mode)
            correct += score
            total += 1
            if log_f:
                write_jsonl(
                    log_f,
                    {
                        "dataset": "triviaqa_rc",
                        "model": model_id,
                        "question": rec["question"],
                        "prompt": rec["prompt"],
                        "golds": rec["golds"],
                        "prediction": out,
                        "score": int(score),
                        "eval_mode": eval_mode,
                    },
                )

    if log_f:
        log_f.close()

    return correct / max(1, total)

def run_ifeval_eval(tokenizer, model, model_id, batch_size, limit, save_path=None):
    ds = load_dataset("google/IFEval", split="train")

    prompts = []
    for item in ds:
        inst = item.get("prompt")
        inst = str(inst).strip()
        if not inst:
            continue

        prompt = f"{inst}\nFollow the instruction precisely.\n"
        prompts.append(prompt)
        if limit > 0 and len(prompts) >= limit:
            break

    outs = []
    for i in range(0, len(prompts), batch_size):
        outs.extend(generate_batch(tokenizer, model, prompts[i : i + batch_size]))

    good = 0
    log_f = open_jsonl(save_path) if save_path else None
    for idx, o in enumerate(outs):
        first = o.strip().splitlines()[0] if o.strip() else ""
        good += 1 if first else 0
        if log_f:
            write_jsonl(
                log_f,
                {
                    "dataset": "ifeval",
                    "model": model_id,
                    "prompt": prompts[idx],
                    "prediction": o,
                    "non_empty_first_line": int(bool(first)),
                },
            )

    if log_f:
        log_f.close()

    return good / max(1, len(outs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results/part1")
    ap.add_argument("--analysis", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit_pokemon", type=int, default=0)
    ap.add_argument("--limit_triviaqa", type=int, default=0)
    ap.add_argument("--limit_ifeval", type=int, default=0)
    ap.add_argument("--pokemon_csv", type=str, default="data/pokemon.csv")
    ap.add_argument("--pokemon_mcq", action="store_true")
    ap.add_argument("--min_per_trait", type=int, default=500)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    bench_path = os.path.join("data", "pokemon_benchmark.csv")

    if args.test:
        model_ids = ["google/gemma-3-1b-it"]
        sizes_b = [1]
    else:
        model_ids = [
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-it",
            "google/gemma-3-12b-it",
        ]
        sizes_b = [1, 4, 12]

    if not args.analysis:
        if not os.path.exists(bench_path):
            ensure_dir(os.path.dirname(bench_path) or ".")
            build_pokemon_benchmark(
                raw_csv_path=args.pokemon_csv,
                out_csv_path=bench_path,
                use_mcq=args.pokemon_mcq,
                min_per_trait=args.min_per_trait,
            )

        rows = []
        responses_dir = os.path.join(args.outdir, "responses")
        ensure_dir(responses_dir)
        
        for mid, sz in zip(model_ids, sizes_b):
            mid_safe = sanitize_filename(mid)
            trait_acc = run_pokemon_eval(
                mid,
                bench_path,
                args.batch_size,
                args.limit_pokemon,
                save_path=(os.path.join(responses_dir, f"pokemon_{mid_safe}.jsonl") if responses_dir else None),
                eval_mode="auto",
            )
            tqa = run_triviaqa_eval(
                mid,
                batch_size=args.batch_size,
                limit=args.limit_triviaqa,
                save_path=(os.path.join(responses_dir, f"triviaqa_rc_{mid_safe}.jsonl") if responses_dir else None),
                eval_mode="contains",
            )
            ife = run_ifeval_eval(
                mid,
                batch_size=args.batch_size,
                limit=args.limit_ifeval,
                save_path=(os.path.join(responses_dir, f"ifeval_{mid_safe}.jsonl") if responses_dir else None),
            )

            row = {"model": mid, "size_b": sz}
            for trait_key, label in TRAIT_PLOT_LABELS.items():
                row[label] = round(trait_acc.get(trait_key, 0.0), 4)
            row["TriviaQA"] = round(tqa, 4)
            row["IFEval"] = round(ife, 4)
            rows.append(row)

        save_table(
            rows,
            os.path.join(args.outdir, "part1_results.csv"),
            os.path.join(args.outdir, "part1_results.json"),
        )

    with open(os.path.join(args.outdir, "part1_results.json")) as f:
        rows = json.load(f)

    sizes = [r["size_b"] for r in rows]
    trait2acc = {
        label: [r[label] for r in rows] for label in TRAIT_PLOT_LABELS.values()
    }
    plot_memorization(
        sizes,
        trait2acc,
        os.path.join(args.outdir, "part1_memorization.png"),
    )

if __name__ == "__main__":
    main()

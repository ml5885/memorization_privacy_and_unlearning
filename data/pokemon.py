import csv
import os
import random
import re
import string
from collections import defaultdict

from tqdm import tqdm

from models.gemma import generate_batch
from utils.part_1 import open_jsonl, write_jsonl

TRAIT_COLUMN_MAP = {
    "Type 1": "type1",
    "HP": "hp",
    "Speed": "speed",
    "Defense": "defense",
}

TRAIT_PLOT_LABELS = {
    "Type 1": "Type1",
    "HP": "HP",
    "Speed": "Speed",
    "Defense": "Defense",
}

RAW_NAME_COL = "name"

def read_pokemon_raw(csv_path):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def build_pokemon_benchmark(raw_csv_path, out_csv_path, use_mcq, min_per_trait):
    data = read_pokemon_raw(raw_csv_path)
    items = []

    pool = {trait: [] for trait in TRAIT_COLUMN_MAP}
    for row in data:
        for trait, col in TRAIT_COLUMN_MAP.items():
            val = (row.get(col) or "").strip()
            if val and val not in pool[trait]:
                pool[trait].append(val)

    for trait, col in TRAIT_COLUMN_MAP.items():
        count = 0
        for row in data:
            name = (row.get(RAW_NAME_COL) or "").strip()
            ans = (row.get(col) or "").strip()
            if not name or not ans:
                continue

            if use_mcq:
                distractors = [x for x in pool[trait] if x != ans][:3]
                letters = ["A", "B", "C", "D"]
                options = [ans] + distractors
                while len(options) < 4:
                    options.append(options[-1])
                
                # Randomly shuffle the options to avoid correct answer always being A
                correct_index = random.randint(0, 3)
                shuffled_options = [None, None, None, None]
                shuffled_options[correct_index] = ans
                
                # Fill in the distractors
                distractor_idx = 0
                for i in range(4):
                    if shuffled_options[i] is None:
                        shuffled_options[i] = distractors[distractor_idx] if distractor_idx < len(distractors) else distractors[-1]
                        distractor_idx += 1
                
                correct_letter = letters[correct_index]
                
                prompt = (
                    f"You are a concise assistant. Respond with only one letter.\n"
                    f"Question: What is the {trait} of {name}?\n"
                    f"Choices: A) {shuffled_options[0]} B) {shuffled_options[1]} C) {shuffled_options[2]} D) {shuffled_options[3]}\n"
                    f"Answer (letter only):"
                )
                items.append({
                    "trait": trait,
                    "name": name,
                    "format": "mcq",
                    "prompt": prompt,
                    "answer": ans,
                    "answer_letter": correct_letter,
                })
            else:
                prompt = (
                    f"Answer using one word only.\n"
                    f"Question: What is the {trait} of {name}?\n"
                    f"Answer (one word):"
                )
                items.append({
                    "trait": trait,
                    "name": name,
                    "format": "short",
                    "prompt": prompt,
                    "answer": ans,
                })

            count += 1
            if count >= min_per_trait:
                break

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0].keys()))
        w.writeheader()
        for it in items:
            w.writerow(it)

def load_pokemon_benchmark(csv_path):
    items = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            items.append(row)
    return items


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
    print(f"\n{'='*80}")
    print(f"Running Pokemon evaluation for {model_id}")
    print(f"{'='*80}")
    examples = load_pokemon_benchmark(bench_path)
    if limit > 0:
        examples = examples[:limit]
    print(f"Total examples: {len(examples)}")

    trait2correct = defaultdict(int)
    trait2total = defaultdict(int)

    log_f = open_jsonl(save_path) if save_path else None

    batch_prompts, batch_meta = [], []
    for ex in tqdm(examples, desc=f"Pokemon eval ({model_id})"):
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

    results = {
        t: trait2correct[t] / max(1, trait2total[t])
        for t in sorted(trait2total.keys())
    }
    print(f"\nPokemon results for {model_id}:")
    for trait, acc in results.items():
        print(f"  {trait}: {acc:.4f}")
    return results

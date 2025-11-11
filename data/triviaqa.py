import re
import string

from datasets import load_dataset
from tqdm import tqdm

from models.gemma import generate_batch
from utils.part_1 import open_jsonl, write_jsonl

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

def run_triviaqa_eval(tokenizer, model, model_id, batch_size, limit, save_path=None, eval_mode: str = "contains"):
    print(f"\n{'='*80}")
    print(f"Running TriviaQA evaluation for {model_id}")
    print(f"{'='*80}")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")

    records = []
    for item in ds:
        q = item["question"]
        golds = [item["answer"]["value"]] + item["answer"].get("aliases", [])
        prompt = f"Question: {q}\nAnswer with just the final answer (no punctuation or explanation):"
        records.append({"prompt": prompt, "golds": golds, "question": q})
        if limit > 0 and len(records) >= limit:
            break
    print(f"Total examples: {len(records)}")

    correct, total = 0, 0
    log_f = open_jsonl(save_path) if save_path else None
    for i in tqdm(range(0, len(records), batch_size), desc=f"TriviaQA eval ({model_id})"):
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

    accuracy = correct / max(1, total)
    print(f"\nTriviaQA results for {model_id}: {accuracy:.4f} ({correct}/{total})")
    return accuracy

import csv
import os

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
                prompt = (
                    f"You are a concise assistant. Respond with only one letter.\n"
                    f"Question: What is the {trait} of {name}?\n"
                    f"Choices: A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}\n"
                    f"Answer (letter only):"
                )
                items.append({
                    "trait": trait,
                    "name": name,
                    "format": "mcq",
                    "prompt": prompt,
                    "answer": options[0],
                    "answer_letter": letters[0],
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

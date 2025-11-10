import csv
import os
import random

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

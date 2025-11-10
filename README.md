# Memorization, Privacy and Unlearning in LLMs

Project code for Programming Assignment 2 for 15-783: Trustworthy AI - Theory & Practice (Fall 2025).

## Repository Structure

Simple overview of the main folders and files:

- `part_1.py` – Runs Part 1 experiments (see below).
  - `data/pokemon.csv` – Original Kaggle Pokémon dataset.
  - `data/pokemon_benchmark.csv` – Generated evaluation benchmark.
  - `data/pokemon.py` – Code to build/load the Pokémon benchmark.
- `models/gemma.py` – Helper to load Gemma models and batch generate text.
- `utils/`
  - `utils/part_1.py` – Utility functions (directory creation, plotting, saving tables, JSONL logging).

## Part 1: Measuring Memorization vs. Model Scale

Make sure that the [Pokemon dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon) from Kaggle is saved at `data/pokemon.csv`.

Run the full experiments with all three Gemma-3 model sizes (1B, 4B, and 12B) by executing:

```bash
python part_1.py
```

This will:

- Build the Pokemon benchmark from the CSV (if it doesn't already exist)
- Run evaluations with `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, and `google/gemma-3-12b-it`
- Evaluate on TriviaQA and IFEval
- Save results to `results/part1_results.csv` and `results/part1_results.json`
- Generate the memorization-vs-scale plot at `results/part1_memorization.png`

## Part 2: Targeted Unlearning

## Part 3: Robustness Evaluation

## Quick Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## References

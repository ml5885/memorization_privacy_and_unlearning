# Memorization, Privacy and Unlearning in LLMs

Project code for Programming Assignment 2 for 15-783: Trustworthy AI - Theory & Practice (Fall 2025).

## Repository Structure

- `part_1.py` – Runs Part 1 experiments (see below).
  - `data/pokemon.csv` – Original Kaggle Pokémon dataset.
  - `data/pokemon_benchmark.csv` – Generated evaluation benchmark.
  - `data/pokemon.py` – Code to build/load the Pokémon benchmark.
- `models/gemma.py` – Helper to load Gemma models and batch generate text.
- `utils/`
  - `utils/part_1.py` – Utility functions (directory creation, plotting, saving tables, JSONL logging).

## Part 1: Measuring Memorization vs. Model Scale

Make sure that the [Pokemon dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon) from Kaggle is saved at `data/pokemon.csv`.

To run evaluations for the three Gemma-3 models on the Pokemon, TriviaQA, and IFEval datasets, run the following commands:

```bash
python part_1.py --model google/gemma-3-1b-it --model_size 1

python part_1.py --model google/gemma-3-4b-it --model_size 4

python part_1.py --model google/gemma-3-12b-it --model_size 12
```

This will:

- Build the Pokemon benchmark from the CSV (if it doesn't already exist)
- Run evaluations with the specified model on the specified dataset(s)
- Evaluate on TriviaQA and IFEval (if included)
- Save results to `results/part1/` directory

### Generating Analysis

To generate the memorization vs. model size plot and accuracy table from existing results, run:

```bash
python part_1.py --analysis
```

## Part 2: Targeted Unlearning

## Part 3: Robustness Evaluation

## Quick Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## References

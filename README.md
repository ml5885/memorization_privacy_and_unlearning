# Memorization, Privacy and Unlearning in LLMs

Project code for Programming Assignment 2 for 15-783: Trustworthy AI - Theory & Practice (Fall 2025).

## Repository Structure

- `part_1.py` – Runs Part 1 experiments (see below).
- `part_2.py` – Part 2 targeted unlearning (Gemma-3-4b-it; held-out Speed; DPO & RMU).
- `data/`
  - `data/pokemon.csv` – Original Kaggle Pokémon dataset.
  - `data/pokemon_benchmark.csv` – Generated evaluation benchmark.
  - `data/pokemon.py` – Code to build/load the Pokémon benchmark.
  - `data/triviaqa.py` – Code to load and evaluate on TriviaQA.
  - `data/ifeval.py` – Code to load and evaluate on IFEval
- `models/gemma.py` – Helper to load Gemma models and batch generate text.
- `unlearning/`
  - `unlearning/ga.py` – Gradient Ascent unlearning
  - `unlearning/gd.py` – Gradient Discent unlearning
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

In Part 2, we perform targeted unlearning on **Gemma-3-4b-it** using the Pokémon MCQ benchmark constructed in Part 1.

To run targeted unlearning on google/gemma-3-4b-it with Gradient Ascent (GA) and Gradient Difference (GD) (LoRA r=32, alpha=64), run:

```bash
# Gradient Ascent (GA)
python part_2.py \
    --algo ga \
    --model google/gemma-3-4b-it \
    --pokemon_csv "data/pokemon.csv" \
    --pokemon_bench "data/pokemon_benchmark_mcq.csv" \
    --outdir "results/part_2/ga" \
    --num_epochs 3 \
    --batch_size 4 \
    --lr 5e-5 \
    --lora_r 32 \
    --lora_alpha 64

# Gradient Difference (GD)
python part_2.py \
    --algo gd \
    --model google/gemma-3-4b-it \
    --pokemon_csv "data/pokemon.csv" \
    --pokemon_bench "data/pokemon_benchmark_mcq.csv" \
    --outdir "results/part_2/gd" \
    --num_epochs 3 \
    --batch_size 4 \
    --lr 5e-5 \
    --lora_r 32 \
    --lora_alpha 64
```

Each run creates a subdirectory under `results/part_2` of the form:

- GA: `results/part_2/ga_pokemon_google__gemma-3-4b-it/ga_google__gemma-3-4b-it`
- GD: `results/part_2/gd_pokemon_google__gemma-3-4b-it/gd_google__gemma-3-4b-it`

## Part 3: Robustness Evaluation

In Part 3, we evaluate the robustness of the unlearning methods from Part 2. For a given model, we:

- Run **prompt-based extraction attacks** (three different prompt templates).
- Run an **optimization-based GCG attack** using nanoGCG.
- Train a **linear probe** on hidden states to test whether forgotten traits remain linearly decodable.

We can run Part 3 robustness evaluation on the GD model (best unlearned) using the following command:

```bash
python part_3.py \
  --model results/part_2/gd_pokemon_google__gemma-3-4b-it/gd_google__gemma-3-4b-it \
  --local_model \
  --batch_size 32 \
  --limit_pokemon 0 \
  --limit_gcg 0 \
  --probe_trait "Type 1" \
  --probe_test_size 50 \
  --probe_batch_size 16 \
  --probe_epochs 20 \
  --probe_lr 1e-2
```

## Quick Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## References

- [nanogcg](https://github.com/GraySwanAI/nanoGCG)

import csv
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from utils.part_1 import ensure_dir, sanitize_filename

def _auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class PokemonForgetExample:
    def __init__(self, trait, name, prompt, answer):
        self.trait = trait
        self.name = name
        self.prompt = prompt
        self.answer = answer

def load_pokemon_forget_examples(csv_path, forget_traits, use_answer_letter=True):
    examples = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            trait = (row.get("trait") or "").strip()
            if trait not in forget_traits:
                continue
            name = (row.get("name") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if use_answer_letter:
                answer = (row.get("answer_letter") or "").strip()
            else:
                answer = (row.get("answer") or "").strip()
            if not prompt or not answer:
                continue
            examples.append(
                PokemonForgetExample(
                    trait=trait,
                    name=name,
                    prompt=prompt,
                    answer=answer,
                )
            )
    return examples

class PromptAnswerDataset(Dataset):
    """Dataset of (prompt, answer) pairs, tokenized for causal LM training."""

    def __init__(self, examples, tokenizer):
        self.input_ids = []
        self.labels = []

        for ex in examples:
            prompt_text = ex.prompt
            answer_text = ex.answer
            if not answer_text.startswith(" "):
                answer_text = " " + answer_text

            prompt_enc = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )
            answer_enc = tokenizer(
                answer_text,
                add_special_tokens=False,
            )

            prompt_ids = prompt_enc["input_ids"]
            answer_ids = answer_enc["input_ids"]

            ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids

            self.input_ids.append(torch.tensor(ids, dtype=torch.long))
            self.labels.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]

def make_collate_fn(pad_token_id):
    def collate(batch):
        input_ids_list, labels_list = zip(*batch)
        max_len = max(t.size(0) for t in input_ids_list)
        bsz = len(input_ids_list)

        input_ids = torch.full(
            (bsz, max_len),
            pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (bsz, max_len),
            -100,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (bsz, max_len),
            dtype=torch.long,
        )

        for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
            L = ids.size(0)
            input_ids[i, :L] = ids
            attention_mask[i, :L] = 1
            labels[i, :L] = lbl

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate

def load_lora_causal_lm(model_id, local_files_only, lora_r=32, lora_alpha=64, lora_dropout=0.0):
    device = _auto_device()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float32
    else:
        dtype = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    base_model.to(device)
    base_model.train()
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base_model, lora_config)
    return tokenizer, model, device

def compute_sequence_logprobs(logits, labels):
    logprobs = torch.log_softmax(logits, dim=-1)

    labels_shifted = labels[:, 1:]
    logprobs_shifted = logprobs[:, :-1, :]

    mask = labels_shifted != -100
    safe_labels = labels_shifted.clone()
    safe_labels[~mask] = 0

    gathered = torch.gather(
        logprobs_shifted,
        dim=2,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    gathered = gathered * mask
    seq_logps = gathered.sum(dim=1)
    return seq_logps

def run_ga_unlearning(
    model_id,
    pokemon_bench_path,
    outdir,
    forget_traits,
    lr=5e-5,
    batch_size=4,
    num_epochs=3,
    local_files_only=False,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.0,
):
    ensure_dir(outdir)

    print("[ga] Loading base model {} with LoRA adapters...".format(model_id))
    tokenizer, model, device = load_lora_causal_lm(
        model_id=model_id,
        local_files_only=local_files_only,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    print("[ga] Model loaded.")

    examples = load_pokemon_forget_examples(
        pokemon_bench_path,
        forget_traits=forget_traits,
        use_answer_letter=True,
    )
    if not examples:
        raise ValueError(
            "No forget examples found in {} for traits {}".format(
                pokemon_bench_path, forget_traits
            )
        )

    print("[ga] Loaded {} forget examples.".format(len(examples)))

    dataset = PromptAnswerDataset(examples, tokenizer)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    model.train()

    total_steps = num_epochs * len(dataloader)
    print(
        "[ga] Starting unlearning: epochs={}, steps per epoch={}, total_steps={}".format(
            num_epochs, len(dataloader), total_steps
        )
    )

    global_step = 0
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logps = compute_sequence_logprobs(outputs.logits, labels)

            loss = logps.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if (step + 1) % 10 == 0 or step == len(dataloader) - 1:
                print(
                    "[ga] epoch={} step={}/{} global_step={} loss={:.4f}".format(
                        epoch + 1, step + 1, len(dataloader), global_step, loss.item()
                    )
                )

    if hasattr(model, "merge_and_unload"):
        print("[ga] Merging LoRA adapter into base model weights.")
        model = model.merge_and_unload()

    adapter_name = "ga_{}".format(sanitize_filename(model_id))
    adapter_dir = os.path.join(outdir, adapter_name)
    ensure_dir(adapter_dir)

    print("[ga] Saving model and tokenizer to {} ...".format(adapter_dir))
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    meta = {
        "algo": "ga",
        "model": model_id,
        "model_dir": adapter_dir,
        "pokemon_bench": pokemon_bench_path,
        "forget_traits": list(forget_traits),
        "lr": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "local_files_only": bool(local_files_only),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "num_examples": len(dataset),
        "total_steps": total_steps,
    }
    meta_path = os.path.join(outdir, "{}_metadata.json".format(adapter_name))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("[ga] Metadata saved to {}".format(meta_path))
    print("[ga] Finished unlearning. Model directory: {}".format(adapter_dir))

    return adapter_dir

import csv
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

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

def load_instruction_retain_examples(dataset_name, split, max_examples):
    ds = load_dataset(dataset_name, split=split)
    examples = []
    for i, row in enumerate(ds):
        if max_examples > 0 and i >= max_examples:
            break
        inst = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        out = (row.get("output") or "").strip()
        if not inst or not out:
            continue
        if inp:
            prompt = inst + "\n\n" + inp
        else:
            prompt = inst
        examples.append(
            PokemonForgetExample(
                trait="retain",
                name="",
                prompt=prompt,
                answer=out,
            )
        )
    return examples

class PromptAnswerDataset(Dataset):

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

def run_gd_unlearning(
    model_id,
    pokemon_bench_path,
    outdir,
    forget_traits,
    retain_dataset_name="tatsu-lab/alpaca",
    retain_split="train",
    retain_max_examples=5000,
    lr=5e-5,
    batch_size=4,
    num_epochs=3,
    lambda_retain=1.0,
    local_files_only=False,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.0,
):
    ensure_dir(outdir)

    print("[gd] Loading base model {} with LoRA adapters...".format(model_id))
    tokenizer, model, device = load_lora_causal_lm(
        model_id=model_id,
        local_files_only=local_files_only,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    print("[gd] Model loaded.")

    forget_examples = load_pokemon_forget_examples(
        pokemon_bench_path,
        forget_traits=forget_traits,
        use_answer_letter=True,
    )
    if not forget_examples:
        raise ValueError(
            "No forget examples found in {} for traits {}".format(
                pokemon_bench_path, forget_traits
            )
        )

    retain_examples = load_instruction_retain_examples(
        retain_dataset_name,
        retain_split,
        retain_max_examples,
    )
    if not retain_examples:
        raise ValueError(
            "No retain examples loaded from dataset {}".format(retain_dataset_name)
        )

    print(
        "[gd] Loaded {} forget examples and {} retain examples.".format(
            len(forget_examples), len(retain_examples)
        )
    )

    forget_dataset = PromptAnswerDataset(forget_examples, tokenizer)
    retain_dataset = PromptAnswerDataset(retain_examples, tokenizer)

    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    forget_loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    retain_loader = DataLoader(
        retain_dataset,
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

    steps_per_epoch = min(len(forget_loader), len(retain_loader))
    total_steps = num_epochs * steps_per_epoch
    print(
        "[gd] Starting unlearning: epochs={}, steps per epoch={}, total_steps={}, lambda_retain={:.3f}".format(
            num_epochs, steps_per_epoch, total_steps, lambda_retain
        )
    )

    for epoch in range(num_epochs):
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)

        for step in range(steps_per_epoch):
            try:
                forget_batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                forget_batch = next(forget_iter)

            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)

            input_ids_f = forget_batch["input_ids"].to(device)
            attention_mask_f = forget_batch["attention_mask"].to(device)
            labels_f = forget_batch["labels"].to(device)

            outputs_f = model(
                input_ids=input_ids_f,
                attention_mask=attention_mask_f,
            )
            logps_f = compute_sequence_logprobs(outputs_f.logits, labels_f)
            loss_forget = logps_f.mean()

            input_ids_r = retain_batch["input_ids"].to(device)
            attention_mask_r = retain_batch["attention_mask"].to(device)
            labels_r = retain_batch["labels"].to(device)

            outputs_r = model(
                input_ids=input_ids_r,
                attention_mask=attention_mask_r,
            )
            logps_r = compute_sequence_logprobs(outputs_r.logits, labels_r)
            loss_retain = -logps_r.mean()

            loss = loss_forget + lambda_retain * loss_retain

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (step + 1) % 10 == 0 or step == steps_per_epoch - 1:
                print(
                    "[gd] epoch={} step={}/{} loss={:.4f} forget={:.4f} retain={:.4f}".format(
                        epoch + 1,
                        step + 1,
                        steps_per_epoch,
                        loss.item(),
                        loss_forget.item(),
                        loss_retain.item(),
                    )
                )

    if hasattr(model, "merge_and_unload"):
        print("[gd] Merging LoRA adapter into base model weights.")
        model = model.merge_and_unload()

    adapter_name = "gd_{}".format(sanitize_filename(model_id))
    adapter_dir = os.path.join(outdir, adapter_name)
    ensure_dir(adapter_dir)

    print("[gd] Saving model and tokenizer to {} ...".format(adapter_dir))
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    meta = {
        "algo": "gd",
        "model": model_id,
        "model_dir": adapter_dir,
        "pokemon_bench": pokemon_bench_path,
        "forget_traits": list(forget_traits),
        "retain_dataset_name": retain_dataset_name,
        "retain_split": retain_split,
        "retain_max_examples": retain_max_examples,
        "lambda_retain": lambda_retain,
        "lr": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "local_files_only": bool(local_files_only),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "num_forget_examples": len(forget_dataset),
        "num_retain_examples": len(retain_dataset),
        "total_steps": total_steps,
    }
    meta_path = os.path.join(outdir, "{}_metadata.json".format(adapter_name))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("[gd] Metadata saved to {}".format(meta_path))
    print("[gd] Finished unlearning. Model directory: {}".format(adapter_dir))

    return adapter_dir

"""
RMU unlearning following the WMDP paper.

References:
- Lennart Heim, Max Nadeau, Neel Nanda, et al.
  "The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning."
  arXiv:2403.03218, 2024. https://arxiv.org/abs/2403.03218
- RMU method: Representation Misdirection for Unlearning (Section 4).
- Reference implementation:
  https://github.com/centerforaisafety/wmdp/tree/main/rmu
"""

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

class PokemonPromptExample(object):
    def __init__(self, trait, name, prompt):
        self.trait = trait
        self.name = name
        self.prompt = prompt

def load_pokemon_prompts(csv_path, traits):
    examples = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            trait = (row.get("trait") or "").strip()
            if trait not in traits:
                continue
            name = (row.get("name") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                continue
            examples.append(
                PokemonPromptExample(
                    trait=trait,
                    name=name,
                    prompt=prompt,
                )
            )
    return examples

class PromptOnlyDataset(Dataset):
    """Dataset of prompts only, for RMU representation-level training."""

    def __init__(self, examples, tokenizer):
        self.input_ids = []
        self.attention_mask = []

        for ex in examples:
            enc = tokenizer(
                ex.prompt,
                add_special_tokens=False,
            )
            ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            mask = torch.ones_like(ids, dtype=torch.long)
            self.input_ids.append(ids)
            self.attention_mask.append(mask)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]

def make_collate_fn(pad_token_id):
    def collate(batch):
        input_ids_list, mask_list = zip(*batch)
        max_len = max(t.size(0) for t in input_ids_list)
        bsz = len(input_ids_list)

        input_ids = torch.full(
            (bsz, max_len),
            pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (bsz, max_len),
            dtype=torch.long,
        )

        for i, (ids, mask) in enumerate(zip(input_ids_list, mask_list)):
            L = ids.size(0)
            input_ids[i, :L] = ids
            attention_mask[i, :L] = mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    return collate

def load_lora_causal_lm(model_id, local_files_only, lora_r=8, lora_alpha=16, lora_dropout=0.0):
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

    # Freeze base model parameters; train only LoRA adapters.
    for param in base_model.parameters():
        param.requires_grad = False

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

def _layer_index_from_config(model, layer_index):
    num_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if num_layers == 0:
        return layer_index
    if layer_index < 0:
        return max(0, num_layers + layer_index)
    return min(layer_index, num_layers)

def run_rmu_unlearning(
    model_id,
    pokemon_bench_path,
    outdir,
    forget_traits,
    retain_traits,
    lr=1e-4,
    batch_size=8,
    num_epochs=1,
    layer_index=4,
    c=6.5,
    alpha=1200.0,
    local_files_only=False,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.0,
):
    """
    Run RMU unlearning on Pokemon prompts.

    Args:
        model_id: Base model ID or local path (e.g., google/gemma-3-4b-it).
        pokemon_bench_path: Path to the MCQ Pokemon benchmark CSV.
        outdir: Directory where the model and metadata will be saved.
        forget_traits: Traits forming the forget distribution.
        retain_traits: Traits forming the retain distribution.
    """
    ensure_dir(outdir)

    print("[rmu] Loading base model %s with LoRA adapters..." % model_id)
    tokenizer, model, device = load_lora_causal_lm(
        model_id=model_id,
        local_files_only=local_files_only,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    print("[rmu] Model loaded.")

    forget_examples = load_pokemon_prompts(
        pokemon_bench_path,
        traits=forget_traits,
    )
    retain_examples = load_pokemon_prompts(
        pokemon_bench_path,
        traits=retain_traits,
    )
    if not forget_examples:
        raise ValueError(
            "No forget examples found in %s for traits %s"
            % (pokemon_bench_path, forget_traits)
        )
    if not retain_examples:
        raise ValueError(
            "No retain examples found in %s for traits %s"
            % (pokemon_bench_path, retain_traits)
        )

    print(
        "[rmu] Loaded %d forget examples and %d retain examples."
        % (len(forget_examples), len(retain_examples))
    )

    forget_dataset = PromptOnlyDataset(forget_examples, tokenizer)
    retain_dataset = PromptOnlyDataset(retain_examples, tokenizer)

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

    model.train()

    # Random direction u is instantiated lazily once we see the first forget batch.
    u = None
    layer_idx = _layer_index_from_config(model, layer_index)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    steps_per_epoch = min(len(forget_loader), len(retain_loader))
    total_steps = num_epochs * steps_per_epoch
    print(
        "[rmu] Starting unlearning: epochs=%d, steps per epoch=%d, total_steps=%d, "
        "layer_index=%d, c=%.3f, alpha=%.1f"
        % (num_epochs, steps_per_epoch, total_steps, layer_idx, c, alpha)
    )

    for epoch in range(num_epochs):
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)

        for step in range(steps_per_epoch):
            forget_batch = next(forget_iter)
            retain_batch = next(retain_iter)

            input_ids_f = forget_batch["input_ids"].to(device)
            attention_mask_f = forget_batch["attention_mask"].to(device)

            outputs_f = model(
                input_ids=input_ids_f,
                attention_mask=attention_mask_f,
                output_hidden_states=True,
            )
            hidden_states_f = outputs_f.hidden_states[layer_idx]  # [B, L, H]

            if u is None:
                hidden_size = hidden_states_f.size(-1)
                u = torch.rand(hidden_size, device=device)
                u = u / u.norm(p=2)

            diff_f = hidden_states_f - c * u
            loss_forget = (diff_f * diff_f).mean()

            input_ids_r = retain_batch["input_ids"].to(device)
            attention_mask_r = retain_batch["attention_mask"].to(device)

            outputs_r = model(
                input_ids=input_ids_r,
                attention_mask=attention_mask_r,
                output_hidden_states=True,
            )
            hidden_updated = outputs_r.hidden_states[layer_idx]

            # Frozen reference hidden states (LoRA disabled)
            with model.disable_adapter():
                with torch.no_grad():
                    frozen_outputs = model(
                        input_ids=input_ids_r,
                        attention_mask=attention_mask_r,
                        output_hidden_states=True,
                    )
                    hidden_frozen = frozen_outputs.hidden_states[layer_idx]

            diff_r = hidden_updated - hidden_frozen
            loss_retain = (diff_r * diff_r).mean()

            loss = loss_forget + alpha * loss_retain

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (step + 1) % 10 == 0 or step == steps_per_epoch - 1:
                print(
                    "[rmu] epoch=%d step=%d/%d loss=%.4f forget=%.4f retain=%.4f"
                    % (
                        epoch + 1,
                        step + 1,
                        steps_per_epoch,
                        loss.item(),
                        loss_forget.item(),
                        loss_retain.item(),
                    )
                )

    adapter_name = "rmu_%s" % sanitize_filename(model_id)
    adapter_dir = os.path.join(outdir, adapter_name)
    ensure_dir(adapter_dir)

    merged_lora = False
    try:
        model = model.merge_and_unload()
        merged_lora = True
        print("[rmu] Merged LoRA adapter into base model weights.")
    except Exception:
        print(
            "[rmu] Warning: merge_and_unload not available or failed;"
            " saving adapter-only model."
        )

    try:
        model.to("cpu")
    except Exception:
        pass

    print("[rmu] Saving model and tokenizer to %s ..." % adapter_dir)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    meta = {
        "algo": "rmu",
        "model": model_id,
        "adapter_dir": adapter_dir,
        "forget_traits": list(forget_traits),
        "retain_traits": list(retain_traits),
        "lr": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "layer_index": layer_idx,
        "c": c,
        "alpha": alpha,
        "local_files_only": bool(local_files_only),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "num_forget_examples": len(forget_dataset),
        "num_retain_examples": len(retain_dataset),
        "total_steps": total_steps,
    "merged_lora": merged_lora,
    }
    meta_path = os.path.join(outdir, "%s_metadata.json" % adapter_name)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("[rmu] Metadata saved to %s" % meta_path)
    print("[rmu] Finished unlearning. Model directory: %s" % adapter_dir)

    return adapter_dir

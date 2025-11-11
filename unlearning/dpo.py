import os
import math

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

MAXLEN = 512

def _tokenize_pair(tokenizer, prompt, response):
    enc = tokenizer(
        prompt + " " + response,
        return_tensors="pt",
        truncation=True,
        max_length=MAXLEN,
    )
    ids = enc["input_ids"][0]
    attn = enc["attention_mask"][0]

    tgt = tokenizer(
        response,
        return_tensors="pt",
        truncation=True,
        max_length=MAXLEN,
    )["input_ids"][0]

    return ids, attn, tgt

def _sum_logprob(model, tokenizer, prompt, response, use_grad):
    ids, attn, tgt = _tokenize_pair(tokenizer, prompt, response)

    ids = ids.unsqueeze(0).to(model.device)
    attn = attn.unsqueeze(0).to(model.device)

    if use_grad:
        out = model(input_ids=ids, attention_mask=attn, labels=ids)
    else:
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn, labels=ids)

    logits = out.logits[0, -tgt.size(0) - 1 : -1, :]
    logp = torch.log_softmax(logits, dim=-1)

    return logp.gather(-1, tgt.to(model.device).unsqueeze(-1)).squeeze(-1).sum()

def _batch_sum_logprob(model, tokenizer, ps, rs, use_grad):
    return torch.stack(
        [
            _sum_logprob(model, tokenizer, p, r, use_grad)
            for p, r in zip(ps, rs)
        ],
        dim=0,
    )

def train_dpo_unlearning(
    tokenizer,
    model,
    ref_model,
    pairs,
    epochs,
    lr,
    beta,
    batch_size,
    save_dir,
    effective_bsz=32,
    weight_decay=0.01,
    warmup_epochs=1,
):
    grad_accum = max(1, math.ceil(effective_bsz / (batch_size * accelerator.num_processes)))
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum)

    model.train()
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    ref_model.eval()
    ref_model.to(next(model.parameters()).device)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def collate(batch):
        ps = [b[0] for b in batch]
        chosen = [b[1] for b in batch]
        rejected = [b[2] for b in batch]
        return ps, chosen, rejected

    dl = DataLoader(
        pairs,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )

    model, ref_model, optimizer, dl = accelerator.prepare(
        model, ref_model, optimizer, dl
    )

    steps_per_epoch = len(dl)
    total_steps = steps_per_epoch * max(1, epochs)
    warmup_steps = steps_per_epoch * max(1, warmup_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for _ in range(epochs):
        for ps, chosen, rejected in tqdm(dl, desc="DPO"):
            with accelerator.accumulate(model):
                lp_c = _batch_sum_logprob(
                    model, tokenizer, ps, chosen, use_grad=True
                )
                lp_r = _batch_sum_logprob(
                    model, tokenizer, ps, rejected, use_grad=True
                )
                lpr_c = _batch_sum_logprob(
                    ref_model, tokenizer, ps, chosen, use_grad=False
                )
                lpr_r = _batch_sum_logprob(
                    ref_model, tokenizer, ps, rejected, use_grad=False
                )

                pref = beta * ((lp_c - lp_r) - (lpr_c - lpr_r))
                loss = -torch.nn.functional.logsigmoid(pref).mean()

                optimizer.zero_grad()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()

    accelerator.wait_for_everyone()
    os.makedirs(save_dir, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    model.eval()
    return accelerator.unwrap_model(model)

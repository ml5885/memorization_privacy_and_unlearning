import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm


def _tokenize_pair(tokenizer, prompt, response):
    text = prompt + " " + response
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"][0]
    attn = enc["attention_mask"][0]
    
    with tokenizer.as_target_tokenizer():
        tgt = tokenizer(response, return_tensors="pt")["input_ids"][0]
    
    return ids, attn, tgt

def _sum_logprob(model, tokenizer, prompt, response):
    ids, attn, tgt = _tokenize_pair(tokenizer, prompt, response)
    ids = ids.unsqueeze(0).to(model.device)
    attn = attn.unsqueeze(0).to(model.device)
    
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, labels=ids)
    
    logits = out.logits[0, -tgt.size(0)-1:-1, :]
    probs = torch.log_softmax(logits, dim=-1)
    
    return probs.gather(-1, tgt.to(model.device).unsqueeze(-1)).squeeze(-1).sum()

def _batch_sum_logprob(model, tokenizer, batch_prompts, batch_responses):
    vals = []
    for p, r in zip(batch_prompts, batch_responses):
        vals.append(_sum_logprob(model, tokenizer, p, r))
    
    return torch.stack(vals, dim=0)

def train_dpo_unlearning(tokenizer, model, ref_model, pairs, epochs, lr, beta, batch_size, save_dir):
    accelerator = Accelerator()
    model.train()
    
    for p in ref_model.parameters():
        p.requires_grad_(False)
    
    optimizer = AdamW(model.parameters(), lr=lr)

    def collate(batch):
        ps = [b[0] for b in batch]
        chosen = [b[1] for b in batch]
        rejected = [b[2] for b in batch]
        return ps, chosen, rejected

    dl = DataLoader(pairs, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=False)
    model, ref_model, optimizer, dl = accelerator.prepare(model, ref_model, optimizer, dl)

    for _ in range(epochs):
        for ps, chosen, rejected in tqdm(dl, desc="DPO"):
            lp_c = _batch_sum_logprob(model, tokenizer, ps, chosen)
            lp_r = _batch_sum_logprob(model, tokenizer, ps, rejected)
            
            with torch.no_grad():
                lpr_c = _batch_sum_logprob(ref_model, tokenizer, ps, chosen)
                lpr_r = _batch_sum_logprob(ref_model, tokenizer, ps, rejected)
            
            pref = beta * ((lp_c - lp_r) - (lpr_c - lpr_r))
            loss = -torch.nn.functional.logsigmoid(pref).mean()
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

    accelerator.wait_for_everyone()
    os.makedirs(save_dir, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    model.eval()
    
    return accelerator.unwrap_model(model)

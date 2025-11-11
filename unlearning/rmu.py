import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

MAXLEN = 512

def _hidden_for_prompts(tokenizer, model, prompts, layer, batch_size):
    hs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]

        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAXLEN,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        h = out.hidden_states[layer]
        idx = enc["attention_mask"].sum(dim=1) - 1
        reps = h[torch.arange(h.size(0), device=h.device), idx, :]
        hs.append(reps.detach().cpu().to(torch.float32))

    return torch.cat(hs, dim=0)

def _pca_topk(X, k):
    X = X - X.mean(dim=0, keepdim=True)

    C = X.T @ X / (X.size(0) - 1)
    U, S, V = torch.linalg.svd(C)

    return U[:, :k].contiguous()

def compute_pca_directions(tokenizer, model, prompts, layer, k, batch_size):
    model.eval()

    X = _hidden_for_prompts(
        tokenizer,
        model,
        prompts,
        layer=layer,
        batch_size=batch_size,
    )
    U = _pca_topk(X, k)
    return U.to(model.device, dtype=torch.float32)

def _tokenize_ce(tokenizer, prompt, response):
    enc = tokenizer(
        prompt + " " + response,
        return_tensors="pt",
        truncation=True,
        max_length=MAXLEN,
    )
    ids = enc["input_ids"][0]
    attn = enc["attention_mask"][0]
    labels = ids.clone()

    return ids, attn, labels

def _proj_loss(h, U):
    h = h.float()
    U = U.float()
    proj = h @ U

    return (proj.pow(2).sum(dim=1)).mean()

def train_rmu_unlearning(
    tokenizer,
    model,
    forget_records,
    refusal,
    U,
    alpha,
    epochs,
    lr,
    batch_size,
    layer,
    save_dir,
):
    accelerator = Accelerator()
    model.train()
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    data = []
    for r in forget_records:
        p = r["prompt"]
        ids, attn, labels = _tokenize_ce(tokenizer, p, refusal)
        data.append((ids, attn, labels))

    def collate(batch):
        ids = [b[0] for b in batch]
        attn = [b[1] for b in batch]
        labels = [b[2] for b in batch]
        maxlen = max(x.size(0) for x in ids)
        pad_id = tokenizer.pad_token_id

        ids_pad = torch.stack(
            [
                torch.nn.functional.pad(
                    x,
                    (0, maxlen - x.size(0)),
                    value=pad_id,
                )
                for x in ids
            ],
            dim=0,
        )
        attn_pad = torch.stack(
            [
                torch.nn.functional.pad(x, (0, maxlen - x.size(0)), value=0)
                for x in attn
            ],
            dim=0,
        )
        labels_pad = ids_pad.clone()

        return {
            "input_ids": ids_pad,
            "attention_mask": attn_pad,
            "labels": labels_pad,
        }

    dl = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    for _ in range(epochs):
        for batch in tqdm(dl, desc="RMU"):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            out = model(**batch, output_hidden_states=True)
            ce = out.loss

            h = out.hidden_states[layer]
            idx = batch["attention_mask"].sum(dim=1) - 1
            reps = h[torch.arange(h.size(0), device=h.device), idx, :]

            rloss = _proj_loss(reps, U)
            loss = ce + alpha * rloss

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

    accelerator.wait_for_everyone()
    os.makedirs(save_dir, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    model.eval()
    return accelerator.unwrap_model(model)

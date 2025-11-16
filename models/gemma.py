import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_gemma_model(model_id, local_files_only=False):
    device = _auto_device()
    tok = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )

    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "left"

    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float32
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    print(f"[info] Using device: {device}")
    return tok, model

def _prepare_inputs(tokenizer, prompts):
    has_template = getattr(tokenizer, "chat_template", None) is not None and hasattr(
        tokenizer, "apply_chat_template"
    )

    if has_template:
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        return tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    else:
        return tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )

def generate_batch(tokenizer, model, prompts):
    device = next(model.parameters()).device
    inputs = _prepare_inputs(tokenizer, prompts)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    padded_input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            min_new_tokens=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for i in range(out.size(0)):
        gen_ids = out[i, padded_input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        results.append(text)
    return results

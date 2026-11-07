"""Pinned dual-head checkpoint; share the original model's frozen backbone storage."""
import threading

import torch
from huggingface_hub import hf_hub_download

from gemma_model import Gemma
from text_normalization import preprocess

MODEL_REPO = "Durrani95/pangram-gemma270-dual-triplet"
MODEL_REVISION = "0df463cf610e21145a0b49091b6c605ccd609089"
STEP = 3841
MAX_TOKENS = 16_384
CLASSES = ("Human", "AI", "Mixed")


def load_experimental(token, base_resources):
    base, tokenizer, _, _ = base_resources
    c = torch.load(hf_hub_download(MODEL_REPO, "checkpoint.pt", revision=MODEL_REVISION,
                                 token=token), map_location="cpu", weights_only=True, mmap=True)
    if c["next_step"] != STEP + 1:
        raise ValueError("Unexpected experimental checkpoint.")
    with torch.device("meta"):
        model = torch.nn.Module()
        model.backbone = Gemma(base.backbone.config, causal=False)
        model.token_head = torch.nn.Linear(base.backbone.config["hidden_size"], 1)
        model.document_head = torch.nn.Linear(base.backbone.config["hidden_size"], 3)
    # Frozen tensors are shared, while every adapter and both heads are independent.
    weights = {name: value for name, value in base.state_dict().items()
               if name.startswith("backbone.") and ".lora_" not in name}
    weights.update({name: value.to(dtype=torch.bfloat16) for name, value in c["model"].items()})
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model, tokenizer, c["settings"].get("preprocessing"), threading.Lock()


@torch.inference_mode()
def predict_experimental(text, resources):
    model, tokenizer, preprocessing, lock = resources
    text = preprocess(text, preprocessing)
    if not text.strip():
        raise ValueError("Please enter some text.")
    inputs = tokenizer(text, return_tensors="pt", truncation=False,
                       return_offsets_mapping=True, return_special_tokens_mask=True)
    count = inputs.input_ids.shape[1]
    if count > MAX_TOKENS:
        raise ValueError(f"This text has {count:,} tokens. Please use at most {MAX_TOKENS:,}; nothing was truncated.")
    valid = inputs.attention_mask.bool() & ~inputs.special_tokens_mask.bool()
    valid_count = int(valid.sum())
    if not valid_count:
        raise ValueError("Please enter text with at least one scorable token.")
    with lock, torch.autocast("cpu", dtype=torch.bfloat16):
        hidden = model.backbone(inputs.input_ids, inputs.attention_mask)
        pooled = (hidden.float() * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp_min(1).unsqueeze(-1)
        doc = model.document_head(pooled).float()
        if valid_count < 256:
            doc[:, 2] = float("-inf")
        probabilities = doc.softmax(1)[0].tolist()
        logits = model.token_head(hidden)[0, :, 0].float()
    ai_token_count = int((logits[valid[0]] >= 0).sum())
    ai_share = ai_token_count / valid_count
    offsets = inputs.offset_mapping[0].tolist()
    usable = [i for i, (a, b) in enumerate(offsets) if b > a and valid[0, i]]
    return {"text": text, "count": count, "valid_count": valid_count,
            "probabilities": probabilities, "ai_share": ai_share, "ai_token_count": ai_token_count,
            "label": CLASSES[max(range(3), key=lambda i: probabilities[i])],
            "offsets": [offsets[i] for i in usable], "logits": logits[usable].tolist()}

"""Pinned four-query share-head checkpoint; shared frozen backbone storage."""
import threading
import math

import torch
from huggingface_hub import hf_hub_download

from gemma_model import Gemma
from text_normalization import preprocess

MODEL_REPO = "Durrani95/pangram-gemma270-share"
MODEL_REVISION = "3b29d72760b3aa75a40034f00a09b092777e166a"
STEP = 3245
MAX_TOKENS = 16_384
CLASSES = ("Human", "AI", "Mixed")


class QueryPool(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.queries = torch.nn.Parameter(torch.empty(4, dim))

    def forward(self, hidden, valid):
        if not valid.any(1).all():
            raise ValueError("No scorable tokens.")
        with torch.autocast(hidden.device.type, enabled=False):
            h = hidden.float()
            scores = torch.einsum("btd,qd->bqt", h, self.queries.float()) / math.sqrt(h.shape[-1])
            weights = scores.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
            attended = torch.einsum("bqt,btd->bqd", weights, h).flatten(1)
            mean = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
            return torch.cat((attended, mean), dim=-1)


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
        model.query_pool = QueryPool(base.backbone.config["hidden_size"])
        model.share_head = torch.nn.Linear(5 * base.backbone.config["hidden_size"], 13)
    # Frozen tensors are shared, while every adapter and both heads are independent.
    weights = {name: value for name, value in base.state_dict().items()
               if name.startswith("backbone.") and ".lora_" not in name}
    weights.update(c["model"])
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
    offsets = inputs.offset_mapping[0].tolist()
    nonempty = inputs.offset_mapping[:, :, 1] > inputs.offset_mapping[:, :, 0]
    valid = inputs.attention_mask.bool() & ~inputs.special_tokens_mask.bool() & nonempty
    valid_count = int(valid.sum())
    if not valid_count:
        raise ValueError("Please enter text with at least one scorable token.")
    with lock, torch.autocast("cpu", dtype=torch.bfloat16):
        hidden = model.backbone(inputs.input_ids, inputs.attention_mask)
        share_logits = model.share_head(model.query_pool(hidden, valid)).float()
        share_probabilities = share_logits.softmax(1)[0]
        bucket = int(share_logits.argmax(1).item())
        verdict = 0 if bucket == 0 else (1 if bucket == 12 else 2)
        probabilities = [float(share_probabilities[0]), float(share_probabilities[12]),
                         float(share_probabilities[1:12].sum())]
        logits = model.token_head(hidden)[0, :, 0].float()
    ai_token_count = int((logits[valid[0]] >= 0).sum())
    ai_share = ai_token_count / valid_count
    offsets = inputs.offset_mapping[0].tolist()
    usable = [i for i, (a, b) in enumerate(offsets) if b > a and valid[0, i]]
    return {"text": text, "count": count, "valid_count": valid_count,
            "probabilities": probabilities, "ai_share": ai_share, "ai_token_count": ai_token_count,
            "label": CLASSES[verdict], "share_bucket": bucket,
            "share_probabilities": share_probabilities.tolist(),
            "offsets": [offsets[i] for i in usable], "logits": logits[usable].tolist()}



def normalize_result(result):
    """Validate a result and recover missing token share from saved token logits."""
    if not isinstance(result, dict) or result.get("label") not in CLASSES:
        raise ValueError("Invalid experimental result; please check the text again.")
    result = dict(result)
    logits = result.get("logits")
    offsets = result.get("offsets")
    if not isinstance(logits, list) or not logits or not isinstance(offsets, list):
        raise ValueError("Missing token predictions; please check the text again.")
    if len(logits) != len(offsets) or not all(math.isfinite(z) for z in logits):
        raise ValueError("Invalid token predictions; please check the text again.")
    if not isinstance(result.get("text"), str):
        raise ValueError("Missing result text.")
    result.setdefault("valid_count", len(logits))
    if result["valid_count"] != len(logits):
        raise ValueError("Inconsistent token count; please check the text again.")
    # Token-derived fraction, not the share-head probability or bucket midpoint.
    result["ai_token_count"] = sum(z >= 0 for z in logits)
    result["ai_share"] = result["ai_token_count"] / len(logits)
    return result

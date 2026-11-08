"""Load checkpoint 5000 once; run the same classifier on CPU."""
import json
import os
import threading

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from gemma_model import GemmaClassifier
from text_normalization import preprocess

MODEL_REPO = "Durrani95/pangram-gemma270-lora-5000"
MODEL_REVISION = "46b0a3298fea688315d22e618ac454a6723700e0"
BASE_REPO = "google/gemma-3-270m"
BASE_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
MAX_TOKENS = 16_384


def load_model(token):
    from pathlib import Path

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    checkpoint = torch.load(
        hf_hub_download(MODEL_REPO, "checkpoint.pt", revision=MODEL_REVISION, token=token),
        map_location="cpu", weights_only=True, mmap=True,
    )
    assert checkpoint["next_step"] == 5001, "Expected checkpoint 5000"
    folder = Path(snapshot_download(
        BASE_REPO, revision=BASE_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer.model"], token=token,
    ))
    config = json.loads((folder / "config.json").read_text())
    # Avoid allocating a second randomly initialized copy of the backbone.
    with torch.device("meta"):
        model = GemmaClassifier(config=config)
    weights = {}
    for shard in folder.glob("*.safetensors"):
        weights.update({"backbone." + name.removeprefix("model."): value
                        for name, value in load_file(str(shard)).items()
                        if name != "lm_head.weight"})
    weights.update(checkpoint["model"])
    model.load_state_dict(weights, strict=True, assign=True)
    model = model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(folder), padding_side="right")
    return model, tokenizer, checkpoint["settings"].get("preprocessing") or "plain-format-v1", threading.Lock()


@torch.inference_mode()
def predict(text, resources):
    model, tokenizer, preprocessing, lock = resources
    text = preprocess(text, preprocessing)
    if not text.strip():
        raise ValueError("Please enter some text.")
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    count = inputs.input_ids.shape[1]
    if count > MAX_TOKENS:
        raise ValueError(f"This text has {count:,} tokens. Please use at most {MAX_TOKENS:,}; nothing was truncated.")
    with lock:
        score = model(inputs.input_ids, inputs.attention_mask).float().sigmoid().item()
    return score, count


def smooth_labels(logits, penalty=2.0):
    """Best binary path: sum logistic losses + penalty per label change."""
    import math
    if penalty < 0 or not math.isfinite(penalty):
        raise ValueError("Use a finite, nonnegative transition penalty.")
    if not logits:
        return []
    costs = [0.0, 0.0]  # No preference for the first label.
    parents = []
    for z in logits:
        if not math.isfinite(z):
            raise ValueError("Non-finite token score.")
        softplus = max(z, 0) + math.log1p(math.exp(-abs(z)))
        emissions = [softplus, softplus - z]
        previous = [min(range(2), key=lambda p: costs[p] + penalty * (p != state))
                    for state in range(2)]
        costs = [costs[previous[s]] + penalty * (previous[s] != s) + emissions[s]
                 for s in range(2)]
        parents.append(previous)
    state = min(range(2), key=lambda s: costs[s])
    labels = []
    for previous in reversed(parents):
        labels.append(state)
        state = previous[state]
    return labels[::-1]


@torch.inference_mode()
def predict_tokens(text, resources):
    """One backbone pass; reuse contextual states for token and document heads."""
    model, tokenizer, preprocessing, lock = resources
    text = preprocess(text, preprocessing)
    if not text.strip():
        raise ValueError("Please enter some text.")
    inputs = tokenizer(text, return_tensors="pt", truncation=False,
                       return_offsets_mapping=True, return_special_tokens_mask=True)
    count = inputs.input_ids.shape[1]
    if count > MAX_TOKENS:
        raise ValueError(f"This text has {count:,} tokens. Please use at most {MAX_TOKENS:,}; nothing was truncated.")
    with lock:
        hidden = model.backbone(inputs.input_ids, inputs.attention_mask)
        logits = model.lm_head(hidden)[0, :, 0].float()
        mask = inputs.attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        document_score = model.lm_head(pooled).float().sigmoid().item()
    offsets = inputs.offset_mapping[0].tolist()
    usable = [i for i, (a, b) in enumerate(offsets)
              if b > a and not inputs.special_tokens_mask[0, i]]
    return {"text": text, "offsets": [offsets[i] for i in usable],
            "logits": logits[usable].tolist(), "count": count,
            "document_score": document_score}


def highlight_tokens(result, labels):
    """Escape user text; offsets refer to the cleaned text actually scored."""
    import html
    import math
    if len(labels) != len(result['offsets']):
        raise ValueError("One label is required per token.")
    parts, cursor = [], 0
    for (start, end), z, label in zip(result['offsets'], result['logits'], labels):
        # Some Unicode characters have multiple tokens with overlapping offsets.
        start = max(start, cursor)
        if end <= start:
            continue
        parts.append(html.escape(result['text'][cursor:start]))
        score = 1 / (1 + math.exp(-max(-80, min(80, z))))
        color, background = ('#000000', '#fee2e2') if label else ('#000000', '#dcfce7')
        title = f"AI-token probability: {score:.1%}"
        parts.append(f'<span title="{title}" style="color:{color};background-color:{background};">'
                     + html.escape(result['text'][start:end]) + '</span>')
        cursor = end
    parts.append(html.escape(result['text'][cursor:]))
    return '<div style="white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.9;color:#000000;background-color:#ffffff;">' + ''.join(parts) + '</div>'

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


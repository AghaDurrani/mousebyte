"""Load checkpoint 4925 once; run the same classifier on CPU."""
import json
import os
import threading

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from qwen_model import QwenClassifier
from text_normalization import preprocess

MODEL_REPO = "Durrani95/pangram-qwen-lora-4925"
MODEL_REVISION = "b17704cb88245aebcfb5e1b388fa44a4ba9bb012"
BASE_REPO = "Qwen/Qwen2.5-0.5B"
BASE_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
MAX_TOKENS = 16_384


def load_model(token):
    from pathlib import Path

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    checkpoint = torch.load(
        hf_hub_download(MODEL_REPO, "checkpoint.pt", revision=MODEL_REVISION, token=token),
        map_location="cpu", weights_only=True, mmap=True,
    )
    assert checkpoint["next_step"] == 4926, "Expected checkpoint 4925"
    folder = Path(snapshot_download(
        BASE_REPO, revision=BASE_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt"],
    ))
    config = json.loads((folder / "config.json").read_text())
    # Avoid allocating a second randomly initialized copy of the backbone.
    with torch.device("meta"):
        model = QwenClassifier(config=config)
    weights = {}
    for shard in folder.glob("*.safetensors"):
        weights.update({"backbone." + name.removeprefix("model."): value
                        for name, value in load_file(str(shard)).items()
                        if name != "lm_head.weight"})
    weights.update(checkpoint["model"])
    model.load_state_dict(weights, strict=True, assign=True)
    model = model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(folder), padding_side="right")
    return model, tokenizer, checkpoint["settings"].get("preprocessing"), threading.Lock()


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

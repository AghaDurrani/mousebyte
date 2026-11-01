"""Shared formatting cleanup. Keep words, equations and citations."""
import re
VERSION = "plain-format-v1"

def _clean(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    tick = chr(96)
    text = re.sub(r"(?m)^[ \t]*(?:" + tick + r"{3,}|~{3,})[^\n]*$", "", text)
    text = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+", "", text)
    # Conservative paired emphasis: preserve (**) equation labels and _____ rules.
    text = re.sub(r"(?<![\w*])\*\*(?=\w)([^*\n]*?\w[.,:;!?]?)\*\*(?![\w*])", r"\1", text)
    text = re.sub(r"(?<![\w_])__(?=[^\W_])([^_\n]*?[^\W_][.,:;!?]?)__(?![\w_])", r"\1", text)
    text = re.sub(tick + r"([^" + tick + r"\n]+)" + tick, r"\1", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)

def normalize(text):
    while True:
        cleaned = _clean(text)
        if cleaned == text:
            return cleaned
        text = cleaned

def preprocess(text, version=None):
    if version is None:
        return text
    if version != VERSION:
        raise ValueError(f"Unknown text preprocessing: {version}")
    return normalize(text)

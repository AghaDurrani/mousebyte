"""Remote GPU inference for the pinned Gemma 1B checkpoint."""
import hashlib
import json
import math
import urllib.error
import urllib.request

MODEL_REVISION = "gemma1b-share-step3600-bf16-v1"
STEP = 3600
MAX_TOKENS = 5000
CLASSES = ("Human", "AI", "Mixed")
DEFAULT_URL = "https://01qo679cm5jw9y-8000.proxy.runpod.net"


def load_experimental(token, base_resources=None, endpoint=DEFAULT_URL):
    """No CPU model loading; derive a service-only credential from existing secrets."""
    if endpoint.rstrip("/") == "https://uwlgpxgwluc1sd-8000.proxy.runpod.net":
        endpoint = DEFAULT_URL
    if not token:
        raise ValueError("Add HF_TOKEN in Streamlit Secrets.")
    if not endpoint.startswith("https://"):
        raise ValueError("Inference URL must use HTTPS.")
    credential = hashlib.sha256(("pangram-inference:" + token).encode()).hexdigest()
    return endpoint.rstrip("/"), credential


def predict_experimental(text, resources):
    if not text.strip():
        raise ValueError("Enter some text first.")
    if len(text) > 120000:
        raise ValueError("Please use a shorter text; maximum is 5,000 tokens.")
    endpoint, credential = resources
    req = urllib.request.Request(
        endpoint + "/predict",
        data=json.dumps({"text": text}).encode(),
        headers={"Authorization": "Bearer " + credential, "Content-Type": "application/json", "User-Agent": "Pangram-Streamlit/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 401:
            raise ValueError("The inference credential does not match. Check the app's HF_TOKEN configuration.") from None
        if error.code in (413, 422, 429):
            try:
                detail = json.loads(error.read()).get("detail")
            except Exception:
                detail = None
            raise ValueError(detail if isinstance(detail, str) else "Input rejected or GPU busy; please try again.") from None
        raise ValueError("The GPU service is offline or starting. Start the inference pod and try again shortly.") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ValueError("The GPU service is offline or starting. Start the inference pod and try again shortly.") from None
    if result.get("model_id") != MODEL_REVISION:
        raise ValueError("The GPU is serving a different model version. Refresh the app configuration.")
    return normalize_result(result)


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

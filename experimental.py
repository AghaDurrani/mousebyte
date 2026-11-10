"""Remote GPU inference for the pinned Gemma 1B checkpoint."""
import hashlib
import json
import math
import urllib.error
import urllib.request

MODEL_REVISION = "gemma1b-share-step3600-bf16-v1"
STEP = 3600
MAX_TOKENS = 5000
# Fixed study policy, calibrated on checkpoint 3600 humans with >64 valid tokens.
AI_THRESHOLD = 0.7730225324630738
MIXED_THRESHOLD = 0.6775493621826173
DECISION_VERSION = "separate-fpr015-over64-v1"
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


def normalize_input_spacing(text):
    """Collapse empty lines; preserve words, nonempty lines and indentation."""
    import re
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n(?:[ \t]*\n)+", "\n", text).strip()


def predict_experimental(text, resources):
    text = normalize_input_spacing(text)
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
    if result["valid_count"] > 64:
        probabilities = result.get("probabilities")
        if (not isinstance(probabilities, list) or len(probabilities) != 3
                or not all(isinstance(p, (int, float)) and math.isfinite(p)
                           and 0 <= p <= 1 for p in probabilities)):
            raise ValueError("Missing or invalid document probabilities.")
        # Order from the service: Human, AI, summed mixed buckets.
        result["label"] = ("AI" if probabilities[1] >= AI_THRESHOLD else
                           "Mixed" if probabilities[2] >= MIXED_THRESHOLD else "Human")
    result["decision_version"] = DECISION_VERSION
    return result

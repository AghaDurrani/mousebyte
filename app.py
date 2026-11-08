"""Run locally: streamlit run app.py"""
import html
import math
import hmac
import os
import time

import streamlit as st

from predict import load_model, predict, predict_tokens, smooth_labels, highlight_tokens
from experimental import load_experimental, predict_experimental, MODEL_REVISION, STEP, normalize_result

RESULT_VERSION = MODEL_REVISION + ":share-result-v2"

st.set_page_config(page_title="Pangram · AI check", page_icon="🔎", layout="centered")


def secret(name):
    if name in os.environ:
        return os.environ[name]
    try:
        return st.secrets.get(name, "")
    except FileNotFoundError:
        return ""


st.title("Text check")
st.write("Paste a passage to try this AI-text detector.")

password = secret("APP_PASSWORD")
if password and not st.session_state.get("authenticated", False):
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Open app")
    if submitted:
        if hmac.compare_digest(entered.encode(), password.encode()):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


@st.cache_resource(show_spinner="Loading your model… The first run takes longer.")
def resources():
    token = secret("HF_TOKEN") or secret("HF_TOKEN_WRITE")
    if not token:
        raise ValueError("Add HF_TOKEN in the app's Secrets settings to load the private model.")
    return load_model(token)



def highlight_pure(result):
    """Document verdict sets hue; raw AI-token score sets shade intensity."""
    is_ai = result["label"] == "AI"
    light, dark = ((254, 226, 226), (185, 28, 28)) if is_ai else ((220, 252, 231), (21, 128, 61))
    parts, cursor = [], 0
    for (start, end), z in zip(result["offsets"], result["logits"]):
        start = max(start, cursor)
        if end <= start:
            continue
        score = 1 / (1 + math.exp(-max(-80, min(80, z))))
        rgb = tuple(round(a + score * (b - a)) for a, b in zip(light, dark))
        foreground = "#ffffff" if score >= 0.65 else "#111827"
        parts.append(html.escape(result["text"][cursor:start]))
        parts.append(
            f'<span title="Raw AI-token score: {score:.1%}" '
            f'style="background-color:rgb{rgb};color:{foreground};">'
            + html.escape(result["text"][start:end]) + '</span>'
        )
        cursor = end
    parts.append(html.escape(result["text"][cursor:]))
    base = "#fee2e2" if is_ai else "#dcfce7"
    return (f'<div style="white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.9;'
            f'background-color:{base};color:#111827;">' + "".join(parts) + "</div>")


def clear_text():
    st.session_state.text = ""


@st.cache_resource(show_spinner="Loading the experimental model…")
def experimental_resources(revision):
    token = secret("HF_TOKEN") or secret("HF_TOKEN_WRITE")
    return load_experimental(token, resources())


document_tab, token_tab, experimental_tab = st.tabs(["Document", "Token highlights", "Experimental"])

with document_tab:
    with st.form("check"):
        text = st.text_area(
            "Your text",
            key="text",
            height=280,
            placeholder="Paste an essay, email, review or article…",
            max_chars=120_000,
        )
        col_check, col_new = st.columns([3, 1])
        with col_check:
            submitted = st.form_submit_button("Check text", type="primary", use_container_width=True)
        with col_new:
            st.form_submit_button("New text", on_click=clear_text, use_container_width=True)

    if submitted:
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            try:
                model = resources()
                start = time.perf_counter()
                with st.spinner("Reading your text…"):
                    score, tokens = predict(text, model)
                st.metric("AI score", f"{score:.1%}")
                st.progress(score)
                st.write("**AI-like**" if score >= .5 else "**Human-like**")
                st.caption(f"{tokens:,} tokens · {time.perf_counter() - start:.1f}s · threshold 0.50")
                if tokens < 64:
                    st.info("Very short passages offer less evidence. Interpret this result cautiously.")
            except ValueError as error:
                st.error(str(error))
            except Exception:
                st.error("The model could not load or finish this request. Check the app logs and Hugging Face access, then try again.")
                # Avoid displaying tracebacks or credentials in the public interface.

with token_tab:
    st.caption("Experimental token scores · red = AI-like · green = human-like")
    with st.form("token_check"):
        token_text = st.text_area("Text to highlight", key="token_text", height=280,
                                  max_chars=120_000)
        token_submitted = st.form_submit_button("Highlight text", type="primary")
    penalty = st.slider("Transition penalty", min_value=0.0, max_value=10.0,
                        value=2.0, step=0.25,
                        help="Higher values discourage label changes. Zero uses each token's raw 0.5 threshold.")
    st.caption("Changing the penalty reuses the token scores—it does not rerun the model.")
    if token_submitted:
        st.session_state.pop("token_result", None)
        try:
            with st.spinner("Reading your text…"):
                st.session_state.token_result = predict_tokens(token_text, resources())
        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error("The model could not finish this request. Check Hugging Face access and try again.")
    result = st.session_state.get("token_result")
    if result is not None:
        labels = smooth_labels(result["logits"], penalty)
        st.markdown(highlight_tokens(result, labels), unsafe_allow_html=True)
        st.caption(f"{result['count']:,} tokens · document AI score {result['document_score']:.1%}")
        st.caption("Highlights show the cleaned text from your last submission. They use the existing classifier on individual tokens; they are not calibrated token probabilities.")

with experimental_tab:
    st.caption(f"Gemma 270M · share-head checkpoint {STEP}")
    with st.form("experimental_check"):
        experimental_text = st.text_area("Your text", key="experimental_text",
                                         height=280, max_chars=120_000)
        experimental_submitted = st.form_submit_button("Check text", type="primary")
    experimental_penalty = st.slider(
        "Highlight smoothing", min_value=0.0, max_value=10.0,
        value=2.0, step=0.25, key="experimental_smoothing",
        help="Higher values discourage switches between Human and AI tokens. Zero turns smoothing off.",
    )
    st.caption("Smoothing changes token highlights and token share only. The document verdict stays unchanged; moving the slider does not rerun the model.")
    if experimental_submitted:
        st.session_state.pop("experimental_result", None)
        try:
            with st.spinner("Reading your text…"):
                result = normalize_result(predict_experimental(experimental_text, experimental_resources(RESULT_VERSION)))
                st.session_state.experimental_result = (RESULT_VERSION, result)
        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error("The experimental model could not load or finish. Check Hugging Face access and the app logs.")
    saved = st.session_state.get("experimental_result")
    if saved and saved[0] != RESULT_VERSION:
        st.session_state.pop("experimental_result", None)
        saved = None
        st.info("The experimental model was updated. Please check your text again.")
    if saved:
        try:
            result = normalize_result(saved[1])
        except (ValueError, KeyError, TypeError):
            st.session_state.pop("experimental_result", None)
            saved = None
            st.info("Please check your text again to refresh this result.")
    if saved:
        st.subheader(result["label"])
        if result["label"] == "Mixed":
            token_labels = smooth_labels(result["logits"], experimental_penalty)
            smoothed_share = sum(token_labels) / len(token_labels)
            ai_col, human_col = st.columns(2)
            ai_col.metric("Smoothed AI-token share", f"{smoothed_share:.1%}")
            human_col.metric("Smoothed human-token share", f"{1-smoothed_share:.1%}")
            st.progress(smoothed_share)
            st.caption(f"Unsmoothed AI-token share: {result['ai_share']:.1%}. Smoothed shares describe the highlighted labels, not verified authorship.")
            st.markdown(highlight_tokens(result, token_labels), unsafe_allow_html=True)
            st.caption("Red = AI-like · green = human-like")
        else:
            st.markdown(highlight_pure(result), unsafe_allow_html=True)
            st.caption("Document verdict sets the colour: green = Human, red = AI. Lighter shades mean lower raw AI-token scores; darker shades mean higher scores.")
        st.caption(f"{result['valid_count']:,} text tokens")
        if result["valid_count"] < 64:
            st.caption("Very short passages offer limited evidence; interpret the result cautiously.")

with st.expander("About this test"):
    st.write("Gemma 3 270M · checkpoint 5000 · CPU inference · up to 16,384 tokens. Your full input is scored together; longer text is rejected rather than silently cut off. Long passages can take a while on CPU.")



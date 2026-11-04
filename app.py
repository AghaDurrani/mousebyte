"""Run locally: streamlit run app.py"""
import hmac
import os
import time

import streamlit as st

from predict import load_model, predict

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


def clear_text():
    st.session_state.text = ""


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

with st.expander("About this test"):
    st.write("Gemma 3 270M · checkpoint 5000 · CPU inference · up to 16,384 tokens. Your full input is scored together; longer text is rejected rather than silently cut off. Long passages can take a while on CPU.")


# Pangram Streamlit tester

Small CPU app using the private checkpoint `Durrani95/pangram-qwen-lora-4925`.
The backbone and checkpoint revisions are pinned. It uses the original custom
architecture and preprocessing, with BF16 weights to keep memory usage down.
No quantization, chunking or automatic truncation. No submitted text is saved.

## Run locally

Use Python 3.11 or 3.12. Install requirements, set `HF_TOKEN`, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Alternatively copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and fill in its values. Never commit secrets.

## Publish for free on Streamlit Community Cloud

1. Put this folder's source files in a GitHub repository.
2. At https://share.streamlit.io select **Create app**, choose that repository,
   branch `main`, entrypoint `app.py`, and Python 3.11 in Advanced settings.
3. Paste the secrets from the example into Advanced settings > Secrets.
   Use an HF read token with access to your private model. Set `APP_PASSWORD`
   if you want the interface password-protected; an obscure URL isn't private.
4. Deploy. Weights download on first use and are cached; cold starts take longer.

The public app does not make the private model or training data public. The
model needs about 1 GB for BF16 parameters, plus Python, tokenization and
activation memory. Long inputs can exceed free-host memory or time limits;
test shorter passages first. The app never upgrades to paid hosting.

## Verification

The model loader requires all checkpoint/backbone keys to match exactly.
Scores are experimental and are not calibrated probabilities of authorship.

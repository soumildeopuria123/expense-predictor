# Deploying the dashboard as a shareable website

Streamlit needs a running Python process, so it cannot be hosted on static /
serverless file hosting. These two hosts run it for free.

Both need the same three things, which are already in this branch:

1. `requirements.txt` — pinned to the versions the 42 tests ran against
   (`pyproject.toml` is not read by Streamlit Community Cloud).
2. `.streamlit/config.toml` — its `allowedHosts` list includes
   `*.streamlit.app` and `*.hf.space`. Without that entry the page returns HTTP
   200 but renders **blank**, because Streamlit rejects the WebSocket handshake.
3. `app/dashboard.py` as the entry point; it adds `src/` to `sys.path` itself,
   so the package does not need to be installed.

## Streamlit Community Cloud (no token, no card)

1. https://share.streamlit.io → **Create app** → *Deploy a public app from GitHub*.
2. Repository `soumildeopuria123/expense-predictor`, branch `deploy/streamlit`,
   main file path `app/dashboard.py`, Python 3.12.
3. **Deploy**.

## Hugging Face Spaces

Create a Space with the **Streamlit** SDK and app file `app/dashboard.py`, then
upload this repo to it (`requirements.txt` and `.streamlit/` included).

## The trained model is not in git

`artifacts/` is gitignored, so a fresh deploy shows the evaluation metrics but
has no model. Press **Train on sample now** on the Score tab: it fits on the
bundled 5,000-row sample in ~80 s and scoring works for the rest of that
container's lifetime. Committing `artifacts/expense_model.joblib` (~12 MB)
removes that first-run step.

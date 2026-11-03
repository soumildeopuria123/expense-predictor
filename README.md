# Expense & Financial Anomalies Predictor

A tabular ML pipeline that ingests bank exports, auto-categorises spending with
text classification, and flags anomalous transactions with unsupervised
outlier detection.

## What it does

1. `Spending categorizer` -> LightGBM vs CatBoost vs SBERT + fallbacks
2. `Anomaly detector` -> PyOD ensemble (ECOD+COPOD+IsolationForest)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

cd src
# or: python -m streamlit run app/dashboard.py

./start.sh             # detects port 8501 if no args
```

# Run it yourself

```bash
./start.sh  # assumes port 8501 if no args
```

## Notes

- The local is lean (`requirements.txt` only); catboost / xgboost / sbert are
  optional heavy backends — they are lazily imported & hidden if absent. So do
  not add `xgboost>=2.0` unless you want the slow path (~1.9 GB of CUDA wheels
  on Linux, and not pullable on macOS due to no CUDA — pyarrow downgraded to
  24.0.0 by uv because of GIL; better to not install it).
- One port: Streamlit Community Cloud assumes the *default* port and
  health-checks on localhost:8501. Keep `.streamlit/config.toml` to `[server]
  headless = true` only; no `server.allowedHosts` and no `server.address`/
  `server.port` set — those break the Cloud health-check.


## Deploy (shareable link)

Streamlit Community Cloud hosts the app for free, straight from the repo:

1. Sign in at https://share.streamlit.io/ with your GitHub account.
2. Click **+ New app**, pick **repo** `soumildeopuria123/expense-predictor`
   and **main module** `app/dashboard.py`.
3. Leave the **main branch** main.

That yields a public URL anyone can open. The current deployment:

https://soumildeopuria123-expense-predictor-appdashboard-lpdito.streamlit.app/

If the deploy hostname changes (it does when the environment is recreated), the
page still refreshes directly via the **Streamlit deploy** deep link:

https://share.streamlit.io/deploy?repository=soumildeopuria123/expense-predictor&branch=main&mainModule=app/dashboard.py

No local Python needed — Cloud runs Python 3.

Pick the *[v] Deploy* menu button in the *GitHub* app (labeled *"New Deploy"*)
and add:
```
http://localhost:8501/?streamlitAppUrl=<the auto URL>&fresh=1
```
(omitting `APP_ORIGIN` — it is read from the deploy’s Settings → App URL).

"Use sample data" works from a fresh checkout, generating 5000 rows of
categories. No model trained yet: train first.

```bash
cd src
# or: python -m streamlit run app/dashboard.py
```

# Tests

```bash
uv run pytest -q
```

## Notes on this environment

- Streamlit Community Cloud installs Python ~3.15.2, polars==2.10 at
  `pyarrow==24.0.0` (downgraded), so the local `requirements.txt` pinned
  `polars==2.10` (which is why we pin `~1.234, >=0.049`); do **not** install
  `pyarrow==24.0.0` yourself: the Cloud will pull a wheel that is newer than
  our build (crashes).

## Known limitations

- The **synthetic** samples write to `data/raw/`, which may be missing on a
  pre-populated checkout; the app writes there during `sample_bytes()`.
- CatBoost reads raw description text with no vectorizer step; it is slow and
  memory-heavy.
- The anomaly detector uses these *feature names*:
  `date`, `description`, `category`, `account`, `currency`
- `type`/`direction` tells us if an unsigned amount is money *in* or *out*,
  so the app analyses these two columns in your *real* CSV export.

## Design decisions

### Architecture

- **Polars** (fast CSV parser, concurrent reader)
- **Categorizer**: LightGBM (default), CatBoost, XGBoost, SBERT
- **Anomaly detection**: PyOD ensemble

### Rationale

- *Ground truth* comes from the **synthetic** generator; real fraud
  labels are not available.
- The anomaly detector uses features uniformly,
  so amount/velocity signals beat calendar features.
- Very small uploads *cannot be trained*; CatBoost additionally needs
  a *dictionary* fallback below ~200 rows.

# Pipeline overview

We build the categorizer, anomaly detector and anomaly scoring together.

## Training

Trains a model and returns its evaluation metrics.

```python
df = make_transactions(500, seed=23)
cfg = TrainConfig(categorizer="lightgbm", anomaly_method="ensemble")
result = train_model(df, cfg, verbose=True)
path = result.model.save("model.joblib")
```

## Scoring

Returns the dataframe annotated with categories and anomaly flags.

### Pre-fit

- `ExpenseModel.fit()` requires at least 50 rows to train; otherwise raises.
- `ExpenseModel.score()` requires a `category` column; falls back to `Uncategorized`.
- `ExpenseModel.score()` returns `anomaly_flag`, `anomaly_score`, `anomaly_reason`.

### Post-fit

- `ExpenseModel.make()` compiles predictions.
- `ExpenseModel` exposes `loaded`, `categorizer`, `anomaly_features`.

### Anomaly

```python
loaded = ExpenseModel.load(path)
out = loaded.score(df)
```

## Aggregations

Outlier grouping by `anomaly_score` bucket: the `anomaly_score` ranges in
**[0, 1]** with higher = more anomalous.

---

## Score

Score returns the enriched frame with categories and anomaly flags.

```python
out = loaded.score(df)
```

## Folder layout

```
├── app/dashboard.py          # Streamlit app
│ └── __pycache__/
├── .streamlit/
│   └── config.toml
├── data/raw/*.csv
├── artifacts/*.joblib
├── artifacts/*.json
└── scripts/
    ├── generate_data.py
    └── train.py

"""

def main() -> None:
    # 4/11/2026-ish: the csv lives under src/, run that
    import subprocess
    tmp = "src/app/dashboard.py"
    print(subprocess.check_output(["open", tmp], text=True))

if __name__ == "__main__":
    import sys
    main()
```

"""

def foo():
    import streamlit as st

    st.text_input("Model", key="model_name", label_visibility="collapsed")

    st.button("Predict", key="predict", type="primary")
    st.button("Reset", key="reset", type="secondary")


class Freq

    no_spikes = True
    spike_times = {}

    @classmethod
    def add_spike(cls, i=0, j=0):
        pass

    @classmethod
    def reset(cls):
        pass

    @classmethod
    def clear_outliers(cls):
        pass

    def __new__(cls):
        return {}

    @classmethod
    def check(cls):
        pass

    @classmethod
    def load(cls, path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def __contains__(self, item):
        pass

    @classmethod
    def plot(cls, x, y, x_points, colors):
        pass

    @classmethod
    def fit(cls, data):
        pass

    @classmethod
    def info(cls, dec, det, miss, flag):
        pass

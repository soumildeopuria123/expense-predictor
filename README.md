# Smart Expense & Financial Anomalies Predictor

A tabular ML pipeline that ingests transaction history (CSV / bank exports),
auto-categorises spending from the text description, and flags anomalous or
fraudulent transactions — plus a Streamlit dashboard for review.

**Stack** (as specified): Polars · LightGBM/CatBoost/XGBoost · PyOD · Streamlit + AgGrid.

## What it does

1. **Ingest** — a flexible CSV loader that matches column headers by alias
   (`date`, `description`, `amount`, `account`, …), parses messy money formats
   (`$1,234.56`, `(45.00)`, `-12.50`, `1 234,56`), and **quarantines** rows that
   don't parse instead of silently dropping them.
2. **Categorise** — text classification on the transaction description with a
   pluggable backend registry.
3. **Detect anomalies** — unsupervised scoring via PyOD, reported with
   calibrated scores and a plain-language reason per flagged row.
4. **Review** — Streamlit dashboard: upload, KPIs, spend breakdown, and an
   interactive AgGrid table for sorting/filtering/click-to-inspect.

## Run it yourself

Requires **Python 3.11+** (verified on 3.12). Every step below was run in a
fresh virtualenv from `pyproject.toml` alone.

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"   # deps + pytest  (~1.9 GB download)

./.venv/bin/python -m pytest -q       # 42 passed
./.venv/bin/python scripts/train.py --rows 5000 --folds 3   # ~80s, writes artifacts/
./start.sh                                              # http://localhost:3000
```

That's the whole loop: install → test → train → dashboard.

`./start.sh` just runs `streamlit run app/dashboard.py` — every server setting
(address, port, allowed hosts) is already in `.streamlit/config.toml`, so the
bare command works too. If it prints *"already running"*, an instance is up
and it prints the URL instead of failing on the port.

Stop a detached instance with `kill "$(cat .streamlit/server.pid)"`.

If the preview hostname changes (it does when the environment is restored) and
the page loads but renders **blank**, Streamlit is rejecting the WebSocket:

```bash
grep 'Rejecting WebSocket' server.log   # shows the host your browser sent
./start.sh <that-host>                  # rewrites allowedHosts for you
```

A running server keeps the allow-list it started with, so restart after
rotating (`kill "$(cat .streamlit/server.pid)" && ./start.sh`);
`start.sh` reminds you if it detects a live instance.

**Notes**

- The install is large (~1.9 GB) because `xgboost>=2.0` hard-depends on
  `nvidia-nccl-cu13` even for CPU-only use. Nothing here needs a GPU.
- The semantic backend is an optional extra — it pulls torch, so it is not in
  the default install:
  `./.venv/bin/pip install -e ".[sbert]"` then choose `sbert` as the
  categorizer. `available_backends()` reports it only when it is importable.
- `artifacts/` is gitignored. Without it the dashboard still loads, but shows
  "No model trained yet" until you run step 3.
- Own data: `./.venv/bin/python scripts/train.py --csv path/to/export.csv`
  (header names are auto-matched; a file with no `category` column is scored,
  not evaluated).
- `server.allowedHosts` guards the **WebSocket** handshake (`/_stcore/stream`),
  not HTTP — a request to `/` returns 200 regardless, so an HTTP-only check can
  look healthy while the app is actually blank. It accepts a list (TOML array,
  or repeat the CLI flag); the `host=a,b` env form does *not* split on commas.
  Each entry is matched exactly, ports ignored; `localhost` must be listed or
  opening it yourself gets 403.

## Use (same steps, for reference)

```bash
# 1. train (writes artifacts/expense_model.joblib + artifacts/metrics.json)
./.venv/bin/python scripts/train.py --rows 5000 --folds 3

# 2. run the dashboard
./start.sh            # or: ./.venv/bin/streamlit run app/dashboard.py

# 3. tests
./.venv/bin/python -m pytest -q        # 42 passed
```

Score your own export without retraining:

```bash
./.venv/bin/python scripts/train.py --csv data/raw/my_export.csv
```

A CSV with no usable `category` column is detected automatically and only
scored — real bank exports carry neither category nor fraud labels, so they can
be scored but not evaluated.

## Results (measured, not aspirational)

On the bundled synthetic dataset (5,041 rows, 17 categories, 2.4% injected
anomalies), chronological 70/30 split:

| Metric | Value |
|---|---|
| Categorisation accuracy | **99.67%** (macro-F1 0.9950) |
| Anomaly average precision | **0.346** vs 0.024 random → **14.1× lift** |
| Precision / Recall / F1 @ operating point | 0.375 / 0.405 / 0.390 |

Per injected-anomaly type (catch rate):

| Type | Caught | Note |
|---|---|---|
| `unknown_merchant` | 6/6 (100%) | strong signal |
| `amount_spike` | 8/14 (57%) | `amount_vs_category` is the single best feature (AP 0.50 alone) |
| `velocity_burst` | 2/13 (15%) | only caught after adding the `txns_prev_2h` short window |
| `odd_hour` | 0/4 (0%) | **known gap** — hour alone is not discriminative at this base rate |

### Honest caveats

- Ground truth is **synthetic**. Real fraud is better hidden than our injected
  patterns; treat these numbers as a comparative baseline between configs, not
  as a production fraud-rate estimate.
- `odd_hour` is undetected at the current operating point. Adding `hour` back to
  the detector made overall AP *worse* (0.360 → 0.251), so it is currently left
  out rather than silently degrading the ensemble.
- Feature selection matters more than the detector here: all 17 engineered
  features scored AP 0.112; the selected 5 score 0.346. IsolationForest picks
  features uniformly at random, so ~14 noise features diluted the 3 informative
  ones (classic unsupervised dilution).

## Design decisions (and why)

- **LightGBM is the default categoriser, not CatBoost.** Measured on the identical
  split: LightGBM 10.0s fit / 0.9934 accuracy vs CatBoost 131.3s / 0.9802 —
  **13× faster and more accurate**. CatBoost native-text also fails on very small
  uploads ("Dictionary size is 0"); there is an explicit dictionary override that
  is applied *only* as a fallback, so real data keeps the fast path.
- **Feature leakage is guarded by tests.** Velocity/context features are computed
  strictly from rows *before* the current timestamp, per account. Sorting must be
  by `(account, time)` — sorting by time alone shatters contiguous runs when
  accounts interleave (this bug was caught by a sentinel-rate test).
- **Tree boosters on sparse text underperform a linear model at small *n***
  (471 train rows: LightGBM 0.54 vs LogReg 0.89) and converge as data grows
  (3,500 rows: 0.9967 vs 0.9977). The out-of-fold estimates are verified against
  an independent holdout rather than assumed correct.

## Layout

```
src/expense_ml/
  ingest.py      # CSV loading, column aliasing, money parsing, quarantine
  synthetic.py   # labelled generator with injected anomaly patterns
  features.py    # leak-free row-local + contextual/velocity features
  categorize.py  # backend registry: lightgbm | catboost | xgboost | sbert
  anomaly.py    # PyOD detectors + combo ensemble voting, thresholds
  pipeline.py    # chronological split, training, metrics, persistence
app/dashboard.py # Streamlit UI
.streamlit/config.toml # address/port/allowedHosts (so no flags are needed)
start.sh         # start the dashboard, rotate the preview hostname
scripts/         # train.py, generate_data.py
tests/           # 42 tests incl. no-future-leakage and anomaly sanity
```

## Notes on this environment

- CPU-only; all models are scikit-learn-compatible estimators, no GPU needed.
- `polars` expressions use `min_samples` (Polars 2.0 removed `min_periods`) and
  `Series.add` / `to_numpy(dtype=...)` no longer exist — the code targets the
  2.x API throughout.

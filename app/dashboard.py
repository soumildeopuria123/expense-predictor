"""Streamlit dashboard for the expense/financial-anomaly predictor.

Layout is driven by the two questions a reviewer actually has:

1. "What does the model do to *my* export?"  -> the Score tab
2. "How well does it actually work?"         -> the Evaluate tab

Anomaly review uses AgGrid so rows can be sorted, filtered and clicked rather
than scanned through a static dataframe.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st

# Run from a plain checkout: the package lives in src/, exactly as
# scripts/ and tests/ bootstrap it.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from expense_ml.anomaly import ALL_METHODS
from expense_ml.categorize import REGISTRY, available_backends
from expense_ml.config import ACCOUNT, AMOUNT, CATEGORY, DATE, DESC, IngestReport
from expense_ml.ingest import IngestError, load_transactions
from expense_ml.pipeline import ExpenseModel, TrainConfig, save_metrics, train_model
from expense_ml.synthetic import ANOMALY_LABEL, make_transactions, write_csv

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "expense_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_transactions.csv"

st.set_page_config(
    page_title="Expense Anomaly Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# caching / resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_model() -> ExpenseModel | None:
    if MODEL_PATH.exists():
        try:
            return ExpenseModel.load(MODEL_PATH)
        except Exception as exc:  # corrupt/stale artifact should not brick the app
            st.warning(f"Saved model could not be loaded ({exc}). Train a new one.")
    return None


@st.cache_data(show_spinner=False)
def ingest(data: bytes, skip_rows: int) -> tuple[pl.DataFrame, IngestReport]:
    return load_transactions(data, skip_rows=skip_rows)


@st.cache_data(show_spinner=False)
def sample_bytes() -> bytes:
    if not SAMPLE_PATH.exists():
        write_csv(make_transactions(5000, seed=7), str(SAMPLE_PATH))
    return SAMPLE_PATH.read_bytes()


def _polars(df: pl.DataFrame):
    return df.to_pandas()


def _flagged_pandas(scored: pl.DataFrame):
    cols = [
        "date", "description", "category", "category_confidence",
        "amount", "account", "anomaly_score", "anomaly_reason",
    ]
    out = scored.filter(pl.col("anomaly_flag")).select(cols)
    return _polars(out.sort("anomaly_score", descending=True))


# --------------------------------------------------------------------------- #
# tabs
# --------------------------------------------------------------------------- #
def score_tab() -> None:
    st.subheader("Score a bank export")
    st.caption(
        "Upload a CSV (date, description, amount columns) or use the bundled "
        "sample. Headers are matched automatically."
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        uploaded = st.file_uploader("CSV export", type=["csv"], key="upl")
    with col_b:
        skip_rows = st.number_input(
            "Skip leading rows", 0, 20, 0,
            help="Some exports carry a bank-name banner before the header row.",
        )
        use_sample = st.button("Use sample data", width="stretch")

    raw: bytes | None = None
    if uploaded is not None:
        raw = uploaded.getvalue()
    elif use_sample or (raw is None and MODEL_PATH.exists()):
        raw = sample_bytes()
    elif raw is None:
        raw = sample_bytes()

    model = get_model()
    if model is None:
        st.info(
            "No trained model yet. Open **Evaluate & Train** and fit one on the "
            "bundled sample (about 2 minutes), then come back."
        )
        if st.button("Train on sample now"):
            st.session_state["go_train"] = True
            st.rerun()
        return

    try:
        df, report = ingest(raw, int(skip_rows))
    except IngestError as exc:
        st.error(str(exc))
        return

    # --- data quality ------------------------------------------------------
    with st.expander(
        f"Data quality — {report.rows_kept:,} of {report.rows_in:,} rows kept",
        expanded=bool(report.rows_dropped),
    ):
        if report.rows_dropped:
            st.warning(
                f"{report.rows_dropped:,} row(s) discarded "
                f"({report.drop_rate:.1%} of the file)."
            )
            for w in report.warnings:
                st.write("• ", w)
            if report.dropped_examples:
                st.caption("Examples:")
                st.dataframe(report.dropped_examples, width="stretch", hide_index=True)
        else:
            st.success("Every row parsed cleanly.")
        st.json(report.resolved_columns)

    # --- score --------------------------------------------------------------
    with st.spinner("Scoring transactions…"):
        t0 = time.time()
        scored = model.score(df)
        elapsed = time.time() - t0

    debits = scored.filter(pl.col(AMOUNT) < 0)
    total_spend = float(debits[AMOUNT].sum())
    flags = scored.filter(pl.col("anomaly_flag"))
    flagged_value = abs(float(flags[AMOUNT].sum())) if flags.height else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Transactions", f"{scored.height:,}")
    k2.metric("Total spend", f"${total_spend:,.2f}")
    k3.metric("Flagged", f"{flags.height:,}", f"{flags.height / max(scored.height, 1):.2%}")
    k4.metric("Flagged value", f"${flagged_value:,.2f}")
    k5.metric("Score time", f"{elapsed:.2f}s")

    # --- breakdown ----------------------------------------------------------
    left, right = st.columns(2)
    with left:
        st.markdown("**Spend by predicted category**")
        by_cat = (
            debits.group_by(CATEGORY)
            .agg(pl.col(AMOUNT).sum().abs().alias("spend"), pl.len().alias("n"))
            .sort("spend", descending=True)
        )
        st.bar_chart(_polars(by_cat).set_index("category")["spend"], height=300)
    with right:
        st.markdown("**Anomaly score distribution**")
        s = scored["anomaly_score"].to_numpy()
        bins = np.linspace(0, 1, 21)
        hist, edges = np.histogram(np.clip(s, 0, 1), bins=bins)
        st.bar_chart(
            {f"{edges[i]:.2f}": int(hist[i]) for i in range(len(hist))},
            height=300,
        )

    # --- AgGrid review -------------------------------------------------------
    st.markdown("**Flagged transactions**")
    if flags.height == 0:
        st.success("Nothing flagged at the current threshold.")
        return

    st.caption("Sort/filter in the grid; select a row for a plain-language reason.")
    gp = _flagged_pandas(scored)
    gp["date"] = gp["date"].astype(str).str.slice(0, 19)
    gp["category_confidence"] = gp["category_confidence"].round(3)
    gp["anomaly_score"] = gp["anomaly_score"].round(4)
    gp["amount"] = gp["amount"].round(2)
    gp = gp.rename(columns={"anomaly_reason": "why_flagged"})
    gp = gp[
        ["date", "description", "category", "category_confidence", "amount",
         "account", "anomaly_score", "why_flagged"]
    ]

    try:
        from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder

        gb = GridOptionsBuilder.from_dataframe(gp)
        gb.configure_pagination(paginationPageSize=25)
        gb.configure_selection("single", use_checkbox=False)
        gb.configure_default_column(
            sortable=True, resizable=True, filter=True, flex=1,
        )
        gb.configure_column("anomaly_score", type="numericColumn", flex=0.8)
        gb.configure_column("amount", type="numericColumn", flex=0.7)
        gb.configure_column("description", flex=1.4)
        gb.configure_column("why_flagged", flex=2.5)
        gb.configure_grid_options(autoSizeStrategy={"type": "fitGridWidth"})
        grid = AgGrid(
            gp,
            gridOptions=gb.build(),
            update_on=["selectionChanged"],
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            allow_unsafe_jscode=False,
            height=420,
            enable_enterprise_modules=False,
            theme="streamlit",
            key="anomaly_grid",
        )
        sel = grid.get("selected_rows")
        if sel is not None and len(sel):
            row = sel[0]
            st.info(
                f"**{row['description']}** — {row['category']} · "
                f"${float(row['amount']):,.2f} · score {float(row['anomaly_score']):.3f}\n\n"
                f"Why: {row['why_flagged']}"
            )
    except Exception as exc:
        st.warning(f"AgGrid unavailable ({exc}); falling back to a plain table.")
        st.dataframe(gp, width="stretch", height=420)


def train_tab() -> None:
    st.subheader("Evaluate & Train")
    st.caption("Metrics come from a chronological 70/30 split, never a random one.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        backend = st.selectbox(
            "Categorizer", available_backends(),
            index=available_backends().index("lightgbm") if "lightgbm" in available_backends() else 0,
            help="LightGBM is the default: measured 10s / 99.3% vs CatBoost "
                 "131s / 98.0% on the same split. CatBoost reads raw text with "
                 "no vectoriser; sbert captures semantics (\"Lyft\"≈\"Uber\").",
        )
    with c2:
        anomaly_method = st.selectbox(
            "Anomaly detector", ALL_METHODS, index=list(ALL_METHODS).index("ensemble"),
            help="ensemble = PyOD detectors voting together (ECOD+COPOD+IsolationForest).",
        )
    with c3:
        contamination = st.slider("Flag rate (contamination)", 0.005, 0.10, 0.03, 0.005)
    with c4:
        n_rows = st.select_slider("Training rows", [1000, 2000, 3000, 5000], value=5000)

    st.markdown(
        "Training runs on the **bundled synthetic dataset** (real exports have no "
        "ground-truth categories or fraud labels). It takes ~2 minutes."
    )
    if st.button("▶ Train", type="primary", disabled=st.session_state.get("training", False)):
        st.session_state["training"] = True
        st.rerun()

    if st.session_state.get("training"):
        status = st.empty()
        progress = st.progress(0.1, text="Preparing…")
        try:
            status.write("Generating dataset…")
            progress.progress(0.2)
            df = make_transactions(int(n_rows), seed=7)
            status.write(
                f"Fitting '{backend}' + '{anomaly_method}' on {df.height:,} rows …"
            )
            progress.progress(0.4)
            cfg = TrainConfig(
                categorizer=backend,
                anomaly_method=anomaly_method,
                contamination=contamination,
            )
            t0 = time.time()
            result = train_model(df, cfg, verbose=True)
            progress.progress(1.0, text=f"Done in {time.time() - t0:.0f}s")
            result.model.save(MODEL_PATH)
            save_metrics(result.metrics, METRICS_PATH)
            st.session_state["metrics"] = result.metrics
            st.session_state["last_train_s"] = time.time() - t0
            get_model.clear()
            status.success(
                f"Trained in {time.time() - t0:.0f}s → artifacts/expense_model.joblib"
            )
        except Exception as exc:
            status.error(f"Training failed: {exc}")
        finally:
            st.session_state["training"] = False
            progress.empty()

    metrics = st.session_state.get("metrics")
    if metrics is None and METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    if metrics is None:
        st.info("No metrics yet — run a training pass above.")
        return
    _render_metrics(metrics)


def _render_metrics(m: dict) -> None:
    cat = m["categorization"]
    an = m["anomaly"]
    st.markdown("### Results")

    # Durable confirmation: the "Trained in Xs" toast lives inside the
    # training block and disappears on the next rerun, so echo it here where
    # it persists alongside the metrics it refers to.
    last_s = st.session_state.get("last_train_s")
    backends = m.get("backends") or {}
    if last_s is not None or backends:
        bits = [f"{b}" for b in (backends.get("categorizer"), backends.get("anomaly")) if b]
        prefix = f"Last training run: {last_s:.0f}s" if last_s is not None else "Current model"
        if bits:
            prefix += " · " + " + ".join(bits)
        st.caption(prefix)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Categorization accuracy", f"{cat['accuracy']:.2%}")
    r2.metric("Macro F1", f"{cat['macro_f1']:.3f}")
    if an.get("ground_truth"):
        r3.metric("Anomaly AP", f"{an['average_precision']:.3f}")
        r4.metric("Lift over random", f"{an['lift']:.1f}×")
    else:
        r3.metric("Flag rate", f"{an['flag_rate']:.2%}")
        r4.metric("Flagged rows", f"{an.get('flagged_rows', 0):,}")

    if an.get("ground_truth"):
        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Precision", f"{an['precision']:.3f}")
        r6.metric("Recall", f"{an['recall']:.3f}")
        r7.metric("F1", f"{an['f1']:.3f}")
        r8.metric("Test base rate", f"{an['test_anomaly_rate']:.2%}",
                  help="random guessing baseline for average precision")
        st.caption(
            "Anomaly AP is a ranking metric: 1.0 is perfect, and the test-set base "
            "rate is what a random ranking scores."
        )

    tabs = st.tabs(["Per-class", "Detection by anomaly type", "Config"])
    with tabs[0]:
        per = cat.get("per_class", {})
        if per:
            import pandas as pd

            tbl = pd.DataFrame(per).T
            tbl.index.name = "category"
            st.dataframe(tbl.style.format("{:.3f}"), width="stretch")
    with tabs[1]:
        by = an.get("detection_by_type") or {}
        if by:
            import pandas as pd

            tbl = pd.DataFrame(by).T
            tbl.index.name = "injected pattern"
            st.dataframe(tbl, width="stretch")
            st.caption(
                "`catch_rate` = share of that pattern scoring in the top 3% of "
                "held-out anomaly scores. `odd_hour` is intentionally weak: the "
                "feature subset favours amount/velocity signals, which measurably "
                "beat the full feature set."
            )
        else:
            st.info("No ground-truth anomaly labels on this dataset.")
    with tabs[2]:
        st.json(m.get("config", {}))
        st.json(m.get("backends", {}))
        st.json({"rows": m.get("rows", {})})


def about_tab() -> None:
    st.subheader("How it works")
    st.markdown(
        """
| Stage | Choice | Why |
|---|---|---|
| Ingestion | **Polars** | Rust-backed, multithreaded, lazy; handles messy bank headers, three amount conventions, and quarantines unparseable rows. |
| Categorisation | **LightGBM** (default), CatBoost, XGBoost, SBERT | Measured: LightGBM 10s/99.3% vs CatBoost 131s/98.0%. CatBoost reads raw text with no TF-IDF step; SBERT understands *"Lyft ride"* ≈ *"Uber trip"*. |
| Anomaly detection | **PyOD** ensemble | Uniform `fit`/`decision_function` across ECOD, COPOD, IsolationForest, LOF, KNN; detectors vote to cut false positives. |
| UI | **Streamlit + AgGrid** | Sortable/filterable/clickable anomaly review instead of a static table. |

        """
    )
    st.markdown(
        """
**Two design decisions worth calling out**

* **No future leakage.** Velocity features are computed with strictly-earlier
  timestamp lookups, not rolling windows over the whole frame. A regression test
  appends a wild future row and asserts every earlier feature is unchanged.
* **Feature selection beats a bigger feature set.** Feeding all 17 features to the
  detector scored AP 0.112 / F1 0.151. The measured 5-feature subset scores
  AP 0.350 / F1 0.400 — because unsupervised detectors choose features uniformly,
  so calendar noise diluted the amount signal that actually carries the signal.
        """
    )
    with st.expander("Known limitations"):
        st.markdown(
            "* Ground truth comes from the **synthetic** generator; real fraud "
            "labels are not available.\n"
            "* `odd_hour` patterns are barely detected by design — the selected "
            "feature subset favours amount/velocity signals.\n"
            "* Very small uploads (<50 rows) cannot train; CatBoost additionally "
            "needs a dictionary fallback below ~30 rows.\n"
            "* Money parsing assumes US thousands separators when only one "
            "separator is present; European readings are used and **reported**."
        )


def main() -> None:
    st.sidebar.title("🛡️ Expense Predictor")
    st.sidebar.caption("Tabular ML · Polars · LightGBM · PyOD")
    choice = st.sidebar.radio(
        "Go to", ["Score transactions", "Evaluate & Train", "How it works"],
        label_visibility="collapsed",
    )
    if st.session_state.pop("go_train", False):
        choice = "Evaluate & Train"
    if choice == "Score transactions":
        score_tab()
    elif choice == "Evaluate & Train":
        train_tab()
    else:
        about_tab()

    model = get_model()
    st.sidebar.divider()
    if model:
        st.sidebar.success(f"Model loaded · {model.categorizer.name}")
        st.sidebar.caption(
            f"Anomaly: {model.anomaly.method} · "
            f"features: {len(model._anomaly_cols)}"
        )
    else:
        st.sidebar.warning("No model trained yet")


main()

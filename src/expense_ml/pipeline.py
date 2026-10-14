"""End-to-end training, evaluation, persistence and scoring.

Data-flow decisions worth stating, because each one is a way this could
silently produce an impressive-but-wrong model:

* **Chronological split.** Transactions are ordered in time; a random split
  would let the model train on September and call August a holdout.
* **Out-of-fold categories for training rows.** The anomaly model consumes the
  *predicted* category, so training rows must be fed out-of-fold predictions.
  Feeding in-sample predictions leaks the label and makes train/test feature
  distributions differ from production.
* **Anomaly features come from a measured subset**, not the full feature set --
  see :data:`expense_ml.features.ANOMALY_FEATURE_COLUMNS`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

from expense_ml.anomaly import AnomalyDetector
from expense_ml.categorize import DEFAULT_BACKEND, build_categorizer
from expense_ml.config import (
    ACCOUNT,
    AMOUNT,
    CATEGORY,
    DATE,
    DESC,
    INCOME_CATEGORIES,
    UNCATEGORIZED,
)
from expense_ml.features import (
    ANOMALY_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FeatureBuilder,
)
from expense_ml.synthetic import ANOMALY_LABEL, ANOMALY_TYPE

DEFAULT_ANOMALY_METHOD = "ensemble"


@dataclass(slots=True)
class TrainConfig:
    categorizer: str = DEFAULT_BACKEND
    anomaly_method: str = DEFAULT_ANOMALY_METHOD
    contamination: float = 0.03
    test_fraction: float = 0.30
    oof_folds: int = 5
    random_state: int = 42
    anomaly_scope: str = "debits"   # debits | all -- what the detector learns from

    def to_dict(self) -> dict[str, Any]:
        return {
            "categorizer": self.categorizer,
            "anomaly_method": self.anomaly_method,
            "contamination": self.contamination,
            "test_fraction": self.test_fraction,
            "oof_folds": self.oof_folds,
            "random_state": self.random_state,
            "anomaly_scope": self.anomaly_scope,
        }


@dataclass(slots=True)
class TrainResult:
    model: "ExpenseModel"
    metrics: dict[str, Any]
    elapsed_s: float


class ExpenseModel:
    """Bundles a categorizer, a feature builder and an anomaly detector."""

    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()
        self.categorizer = build_categorizer(self.config.categorizer)
        self.features = FeatureBuilder()
        self.anomaly = AnomalyDetector(
            method=self.config.anomaly_method,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        )
        self.category_labels: list[str] = []
        self.is_fitted = False

    # ------------------------------------------------------------------ train
    def fit(self, df: pl.DataFrame, *, verbose: bool = False) -> "ExpenseModel":
        if df.height < 50:
            raise ValueError(
                f"Need at least 50 rows to train; got {df.height}."
            )
        cfg = self.config

        if CATEGORY not in df.columns:
            raise ValueError(
                "Training requires a 'category' column. Bank exports rarely "
                "include one -- train on the bundled sample, or supply a "
                "labelled CSV."
            )

        # Sort first, then read the label column. Capturing labels before the
        # sort would leave them in input order while the text follows the new
        # order -- invisible on pre-sorted synthetic data, catastrophic on a
        # shuffled bank export.
        df = df.sort([DATE, "transaction_id"])
        labels = df[CATEGORY]
        has_labels = labels.dtype == pl.String and labels.is_not_null().any() \
            and labels.n_unique() > 1
        if not has_labels:
            raise ValueError(
                "Training requires a 'category' column with more than one "
                "distinct value. Bank exports rarely include it -- train on the "
                "bundled sample, or supply a labelled CSV."
            )

        from sklearn.preprocessing import LabelEncoder

        cut = max(50, int(round(df.height * (1.0 - cfg.test_fraction))))
        cut = min(cut, df.height - 10)
        train, test = df.head(cut), df.tail(df.height - cut)

        y_all = labels.to_numpy()
        self.category_labels = sorted({str(v) for v in y_all if v is not None})
        le = LabelEncoder().fit(self.category_labels)
        self._label_encoder = le

        texts = df[DESC].to_list()
        y = le.transform(y_all).astype(np.int64)

        if verbose:
            print(f"[1/4] fitting categorizer '{cfg.categorizer}' on {cut} rows ...", flush=True)
        t0 = time.time()
        self.categorizer.fit([texts[i] for i in range(cut)], y[:cut])
        if verbose:
            print(f"      done in {time.time()-t0:.1f}s", flush=True)

        # Out-of-fold predictions for training rows; in-sample for the holdout.
        if verbose:
            print(f"[2/4] out-of-fold categories ({cfg.oof_folds} folds) ...", flush=True)
        oof = self.categorizer.predict_oof(
            [texts[i] for i in range(cut)], y[:cut], n_splits=cfg.oof_folds
        )
        train = train.with_columns(
            pl.Series(le.inverse_transform(oof)).alias(CATEGORY)
        )
        test_cat = self.categorizer.predict([texts[i] for i in range(cut, df.height)])
        test = test.with_columns(pl.Series(le.inverse_transform(test_cat)).alias(CATEGORY))

        if verbose:
            print(f"[3/4] building features ...", flush=True)
        Xtr = self.features.fit_transform(train)
        self.features.record_train_stats(Xtr)
        Xte = self.features.transform(test, use_history=True)

        if verbose:
            print(f"[4/4] fitting anomaly detector '{cfg.anomaly_method}' ...", flush=True)
        scope_mask = self._scope_mask(train)
        if not scope_mask.any():
            scope_mask = np.ones(train.height, dtype=bool)
        cols = [FEATURE_COLUMNS.index(c) for c in ANOMALY_FEATURE_COLUMNS]
        self.anomaly.fit(Xtr[scope_mask][:, cols])
        self._anomaly_cols = cols
        self.is_fitted = True

        metrics = self._evaluate(train, test, Xtr, Xte, y[:cut], y[cut:], le)
        return metrics

    def _scope_mask(self, df: pl.DataFrame) -> np.ndarray:
        if self.config.anomaly_scope == "all":
            return np.ones(df.height, dtype=bool)
        return df["is_debit"].to_numpy() if "is_debit" in df.columns else (
            (df[AMOUNT] < 0).to_numpy()
        )

    # --------------------------------------------------------------- evaluate
    def _evaluate(
        self,
        train: pl.DataFrame,
        test: pl.DataFrame,
        Xtr: np.ndarray,
        Xte: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        le,
    ) -> dict[str, Any]:
        # --- categorisation ------------------------------------------------
        texts_test = test[DESC].to_list()
        pred = self.categorizer.predict(texts_test)
        proba = self.categorizer.predict_proba(texts_test)
        rep = classification_report(
            y_test, pred, target_names=[str(c) for c in le.classes_],
            output_dict=True, zero_division=0,
        )

        # --- anomaly -------------------------------------------------------
        cols = self._anomaly_cols
        scope = self._scope_mask(test)
        s = self.anomaly.score(Xte[:, cols])
        flags = self.anomaly.labels(Xte[:, cols])

        metrics: dict[str, Any] = {
            "config": self.config.to_dict(),
            "rows": {"train": train.height, "test": test.height},
            "categorization": {
                "accuracy": float(accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
                "macro_precision": float(rep["macro avg"]["precision"]),
                "macro_recall": float(rep["macro avg"]["recall"]),
                "per_class": {
                    str(c): {
                        "precision": round(rep[str(c)]["precision"], 4),
                        "recall": round(rep[str(c)]["recall"], 4),
                        "f1": round(rep[str(c)]["f1-score"], 4),
                        "support": int(rep[str(c)]["support"]),
                    }
                    for c in le.classes_
                },
            },
            "anomaly": {},
            "backends": {
                "categorizer": self.categorizer.name,
                "anomaly": self.config.anomaly_method,
                "anomaly_features": list(ANOMALY_FEATURE_COLUMNS),
            },
        }

        if ANOMALY_LABEL in test.columns and test[ANOMALY_LABEL].dtype == pl.Boolean:
            y_true = test[ANOMALY_LABEL].to_numpy()
            P, R, F, _ = precision_recall_fscore_support(
                y_true, flags, average="binary", zero_division=0
            )
            metrics["anomaly"] = {
                "ground_truth": True,
                "test_anomaly_rate": float(y_true.mean()),
                "flag_rate": float(flags.mean()),
                "precision": float(P),
                "recall": float(R),
                "f1": float(F),
                "average_precision": float(average_precision_score(y_true, s)),
                "random_baseline_ap": float(y_true.mean()),
                "lift": float(average_precision_score(y_true, s) / max(y_true.mean(), 1e-9)),
                "detection_by_type": self._by_type(test, s, flags),
            }
        else:
            metrics["anomaly"] = {
                "ground_truth": False,
                "flag_rate": float(flags.mean()),
                "flagged_rows": int(flags.sum()),
                "note": "No injected-anomaly labels available; precision/recall "
                        "cannot be computed for this file.",
            }
        return metrics

    @staticmethod
    def _by_type(test: pl.DataFrame, scores: np.ndarray, flags: np.ndarray) -> dict[str, dict]:
        if ANOMALY_TYPE not in test.columns:
            return {}
        typ = test[ANOMALY_TYPE].to_numpy()
        thr = np.quantile(scores, 1.0 - 0.03)
        out: dict[str, dict] = {}
        for t in sorted({str(v) for v in typ if v}):
            m = typ == t
            if not m.any():
                continue
            out[t] = {
                "n": int(m.sum()),
                "caught": int((scores[m] >= thr).sum()),
                "catch_rate": float((scores[m] >= thr).mean()),
                "mean_score": round(float(scores[m].mean()), 4),
            }
        return out

    # ----------------------------------------------------------------- score
    def score(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return the frame annotated with categories and anomaly findings."""
        if not self.is_fitted:
            raise RuntimeError("ExpenseModel.score called before fit.")
        work = df.sort([DATE, "transaction_id"])
        cats = self.categorizer.predict(work[DESC].to_list())
        proba = self.categorizer.predict_proba(work[DESC].to_list())
        labels = self._label_encoder.inverse_transform(cats)
        confidence = proba.max(axis=1)

        # category is a *feature*, so it must be assigned before transforming
        work = work.with_columns(
            pl.Series(labels).alias("_pred_category"),
            pl.Series(confidence).alias("category_confidence"),
        )
        X = self.features.transform(work, use_history=True)

        cols = self._anomaly_cols
        raw = self.features.deviations(
            X[:, cols], columns=ANOMALY_FEATURE_COLUMNS, top_k=len(cols)
        )
        scores = self.anomaly.score(X[:, cols])
        flags = self.anomaly.labels(X[:, cols])

        reasons = []
        for row in raw:
            parts = []
            for name, z in row:
                direction = "unusually high" if z > 0 else "unusually low"
                parts.append(f"{name.replace('_', ' ')} {direction}")
            reasons.append("; ".join(parts))

        return work.with_columns(
            pl.Series(labels).alias(CATEGORY),
            pl.Series(scores).alias("anomaly_score"),
            pl.Series(flags).alias("anomaly_flag"),
            pl.Series(reasons).alias("anomaly_reason"),
        )

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self.config.to_dict(),
                "categorizer": self.categorizer,
                "features": self.features,
                "anomaly": self.anomaly,
                "category_labels": self.category_labels,
                "label_encoder": getattr(self, "_label_encoder", None),
                "anomaly_cols": getattr(self, "_anomaly_cols", None),
                "version": 1,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExpenseModel":
        blob = joblib.load(Path(path))
        cfg = TrainConfig(**blob["config"])
        model = cls(cfg)
        model.categorizer = blob["categorizer"]
        model.features = blob["features"]
        model.anomaly = blob["anomaly"]
        model.category_labels = blob["category_labels"]
        model._label_encoder = blob["label_encoder"]
        model._anomaly_cols = blob["anomaly_cols"]
        model.is_fitted = True
        return model


def train_model(
    df: pl.DataFrame,
    config: TrainConfig | None = None,
    *,
    verbose: bool = False,
) -> TrainResult:
    """Fit a model and return it alongside its evaluation metrics."""
    t0 = time.time()
    model = ExpenseModel(config or TrainConfig())
    metrics = model.fit(df, verbose=verbose)
    return TrainResult(model=model, metrics=metrics, elapsed_s=time.time() - t0)


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str))
    return path

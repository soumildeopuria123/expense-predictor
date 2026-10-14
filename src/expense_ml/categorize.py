"""Spending categorisation from transaction descriptions.

A small registry of interchangeable backends behind one interface, so the
trade-off between them is *measurable* rather than asserted:

* ``catboost`` (default) -- consumes raw text natively, no vectoriser step.
* ``lightgbm`` / ``xgboost`` -- TF-IDF word + character n-grams.
* ``sbert`` -- frozen sentence-transformer embeddings plus a linear classifier,
  which is the only backend that knows "Lyft ride" and "Uber trip" are the same
  kind of thing despite sharing no tokens.

All backends take integer-encoded labels and expose ``classes_`` as that same
integer array; :mod:`expense_ml.pipeline` owns the human-readable mapping.

Two CatBoost specifics cost real debugging time and are encoded here:

* ``text_features`` must be declared on the *model*;
* the dictionary config must override ``occurrence_lower_bound`` to ``"1"``,
  otherwise Catboost drops the entire vocabulary on small uploads and raises
  ``Dictionary size is 0`` below roughly 50 rows.

Note also that every CatBoost text-option value is a *string*, including
numeric ones -- passing ints raises "Can't parse parameter".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

RANDOM_STATE = 42


class BaseCategorizer(ABC):
    """Common interface: fit, predict_proba, predict, and leak-safe OOF."""

    name: str = "base"

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.classes_: np.ndarray | None = None

    @abstractmethod
    def _fit_fold(self, texts: Sequence[str], y: np.ndarray) -> None:
        """Fit a fresh model; called once per fold and once on all data."""

    @abstractmethod
    def _predict_proba_fold(self, texts: Sequence[str]) -> np.ndarray:
        """Probability matrix with columns ordered by that fold's ``classes_``."""

    @property
    @abstractmethod
    def fold_classes_(self) -> np.ndarray:
        """Classes the currently fitted fold model knows about."""

    # -- public API -----------------------------------------------------------
    def fit(self, texts: Sequence[str], y: np.ndarray) -> "BaseCategorizer":
        """Fit on ``y``, remapping labels onto ``0..n-1`` first.

        Backends disagree about what they accept. Catboost validates
        ``classes_count`` against the *largest* label, so passing
        ``[0, 1, 6]`` with ``classes_count=3`` dies with "Value in target (6) is
        greater than specified class count" -- a confusing error that only shows
        up when callers supply their own labels rather than a LabelEncoder's.
        Remapping here keeps every backend's contract identical.
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) == 0:
            raise ValueError("Cannot fit on an empty label array.")
        y_enc = np.searchsorted(self.classes_, y).astype(np.int64)
        self._fit_fold(texts, y_enc)
        self._fold_classes = self.classes_
        return self

    def clone(self) -> "BaseCategorizer":
        """Fresh estimator of the same type and hyper-parameters.

        Used to build out-of-fold models without disturbing this instance.
        ``__new__``-style shallow copies are fragile: they carry fitted state
        across folds, which is exactly what CV must not do.
        """
        import inspect

        params = {
            n: getattr(self, n)
            for n in inspect.signature(self.__init__).parameters
            if n != "self" and hasattr(self, n)
        }
        return type(self)(**params)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("predict_proba called before fit")
        return self._aligned_proba(texts, self.classes_)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(texts), axis=1)]

    def _aligned_proba(self, texts: Sequence[str], global_classes: np.ndarray) -> np.ndarray:
        """Reindex a fold model's probabilities onto the global class order."""
        raw = self._predict_proba_fold(texts)
        known = getattr(self, "_fold_classes", None)
        if known is None:
            known = self.fold_classes_
        cols = {c: i for i, c in enumerate(known)}
        out = np.zeros((raw.shape[0], len(global_classes)), dtype=np.float64)
        for j, c in enumerate(global_classes):
            i = cols.get(c)
            if i is not None:
                out[:, j] = raw[:, i]
        # guard against a fold that predicted nothing at all
        totals = out.sum(axis=1, keepdims=True)
        bad = totals[:, 0] <= 0
        if bad.any():
            out[bad] = 1.0 / len(global_classes)
        return out

    def predict_oof(
        self,
        texts: Sequence[str],
        y: np.ndarray,
        n_splits: int = 5,
    ) -> np.ndarray:
        """Out-of-fold class predictions.

        Used to build anomaly features for the *training* rows. Fitting the
        anomaly model on categories predicted in-sample would leak the label
        straight through, and the train/test feature distributions would then
        differ from what production sees.
        """
        from sklearn.model_selection import StratifiedKFold

        y = np.asarray(y)
        global_classes = np.unique(y)
        counts = np.bincount(y, minlength=len(global_classes))
        folds = int(max(2, min(n_splits, int(counts.min()))))
        if folds < 2:
            # Too few examples per class for stratified CV; the caller falls
            # back to in-sample predictions and should warn.
            return self.predict(texts)

        oof = np.zeros(len(y), dtype=global_classes.dtype)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        for tr_idx, val_idx in skf.split(np.zeros(len(y)), y):
            fold = self.clone()
            fold.fit([texts[i] for i in tr_idx], y[tr_idx])
            proba = fold._aligned_proba([texts[i] for i in val_idx], global_classes)
            oof[val_idx] = global_classes[np.argmax(proba, axis=1)]
        return oof

    def get_params(self) -> dict:
        return {"name": self.name, "random_state": self.random_state}


# ----------------------------------------------------------------------------- #
# CatBoost: native text
# ----------------------------------------------------------------------------- #
class CatBoostCategorizer(BaseCategorizer):
    name = "catboost"

    def __init__(
        self,
        iterations: int = 150,
        depth: int = 6,
        learning_rate: float = 0.1,
        eval_fraction: float = 0.2,
        od_wait: int = 25,
        random_state: int = RANDOM_STATE,
        text_occurrence_lower_bound: int = 1,
        max_dictionary_size: int = 50000,
    ):
        super().__init__(random_state)
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.eval_fraction = eval_fraction
        self.od_wait = od_wait
        self.text_occurrence_lower_bound = text_occurrence_lower_bound
        self.max_dictionary_size = max_dictionary_size
        self._model = None

    def _dictionaries(self) -> list[dict]:
        # Both BiGram and Word are required: CatBoost's default feature_calcers
        # reference the BiGram dictionary by id, and omitting it raises
        # "No options for dictionary_id BiGram".
        base = {
            "occurrence_lower_bound": str(self.text_occurrence_lower_bound),
            "max_dictionary_size": str(self.max_dictionary_size),
            "token_level_type": "Word",
            "start_token_id": "0",
            "skip_step": "0",
            "end_of_word_token_policy": "Insert",
            "end_of_sentence_token_policy": "Skip",
        }
        return [
            {"dictionary_id": "BiGram", "gram_order": "2", **base},
            {"dictionary_id": "Word", "gram_order": "1", **base},
        ]

    @staticmethod
    def _frame(texts: Sequence[str]):
        import pandas as pd

        return pd.DataFrame({"description": list(texts)})

    @property
    def fold_classes_(self) -> np.ndarray:
        return np.asarray(self._model.classes_)

    def _fit_fold(self, texts: Sequence[str], y: np.ndarray) -> None:
        from catboost import CatBoostClassifier, CatBoostError, Pool

        frame = self._frame(texts)
        n_classes = len(np.unique(y))
        common = dict(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function="MultiClass",
            classes_count=n_classes,
            random_seed=self.random_state,
            verbose=0,
            allow_writing_files=False,
            # Oversubscription is a net loss on this small machine: measured
            # 3.1s at thread_count=-1 versus 1.5s at 1 for the same fit.
            thread_count=1,
            text_features=["description"],
            # Text features are ~370ms/iteration on two cores, so the iteration
            # count is only a ceiling: early stopping on a held-out fraction
            # typically converges around 20-30 trees (measured 13.6s rather
            # than the 150s an uncapped 400-iteration fit would take).
            eval_fraction=self.eval_fraction,
            od_type="Iter",
            od_wait=self.od_wait,
        )
        pool = Pool(frame, label=y, text_features=["description"])
        try:
            self._model = CatBoostClassifier(**common)
            self._model.fit(pool)
        except CatBoostError as exc:
            # Very small uploads can leave CatBoost's default dictionaries with
            # no vocabulary. Supplying an explicit dictionary config fixes it,
            # but costs ~6x in fit time, so it is a fallback rather than the
            # default path.
            if "Dictionary size is 0" not in str(exc):
                raise
            self._model = CatBoostClassifier(**common, dictionaries=self._dictionaries())
            self._model.fit(pool)

    def _predict_proba_fold(self, texts: Sequence[str]) -> np.ndarray:
        from catboost import Pool

        return np.asarray(
            self._model.predict_proba(Pool(self._frame(texts), text_features=["description"]))
        )


# ----------------------------------------------------------------------------- #
# TF-IDF + gradient boosting
# ----------------------------------------------------------------------------- #
class TfidfBoosterCategorizer(BaseCategorizer):
    """Shared implementation for the LightGBM and XGBoost backends."""

    booster: str = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 350,
        learning_rate: float = 0.06,
        max_depth: int = -1,
        max_features_word: int = 40000,
        max_features_char: int = 30000,
        min_df: int = 2,
        random_state: int = RANDOM_STATE,
        n_jobs: int = 2,
    ):
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features_word = max_features_word
        self.max_features_char = max_features_char
        self.min_df = min_df
        self.n_jobs = n_jobs
        self._vectorizer = None
        self._clf = None

    def _build_vectorizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=self.min_df,
                        sublinear_tf=True,
                        max_features=self.max_features_word,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=self.min_df,
                        sublinear_tf=True,
                        max_features=self.max_features_char,
                    ),
                ),
            ]
        )

    def _build_clf(self):
        if self.booster == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
            )
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=8 if self.max_depth < 0 else self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.9,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="mlogloss",
        )

    @property
    def fold_classes_(self) -> np.ndarray:
        return np.asarray(self._clf.classes_)

    def _fit_fold(self, texts: Sequence[str], y: np.ndarray) -> None:
        self._vectorizer = self._build_vectorizer()
        self._clf = self._build_clf()
        X = self._vectorizer.fit_transform(list(texts))
        self._clf.fit(X, y)

    def _predict_proba_fold(self, texts: Sequence[str]) -> np.ndarray:
        X = self._vectorizer.transform(list(texts))
        return np.asarray(self._clf.predict_proba(X))


class LightGBMCategorizer(TfidfBoosterCategorizer):
    name = "lightgbm"
    booster = "lightgbm"


class XGBoostCategorizer(TfidfBoosterCategorizer):
    name = "xgboost"
    booster = "xgboost"


# ----------------------------------------------------------------------------- #
# Sentence-transformer embeddings + linear head
# ----------------------------------------------------------------------------- #
class SentenceTransformerCategorizer(BaseCategorizer):
    name = "sbert"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        C: float = 8.0,
        random_state: int = RANDOM_STATE,
    ):
        super().__init__(random_state)
        self.model_name = model_name
        self.C = C
        self._encoder = None
        self._clf = None

    @staticmethod
    def available() -> bool:
        """Is the optional sbert stack installed?

        Deliberately uses ``find_spec`` rather than importing. The eager
        ``import sentence_transformers`` adds ~4,000 modules (torch + the
        whole transformers tree) to the process, and Streamlit's file watcher
        walks every loaded package to decide what to hot-reload -- logging
        100+ ``ModuleNotFoundError: torchvision`` tracebacks, since the image
        modules in transformers want a dependency we do not install. Answering
        "is it installed?" should not cost an import. The real import stays
        lazy in ``_encode()``, so a broken install still fails at use time
        with the factory's explanatory message.
        """
        from importlib.util import find_spec

        try:
            return find_spec("sentence_transformers") is not None
        except Exception:
            return False

    def _encode(self, texts: Sequence[str]):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return np.asarray(
            self._encoder.encode(
                list(texts), normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float64,
        )

    @property
    def fold_classes_(self) -> np.ndarray:
        return np.asarray(self._clf.classes_)

    def _fit_fold(self, texts: Sequence[str], y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        self._clf = LogisticRegression(
            max_iter=2000, C=self.C, solver="lbfgs", random_state=self.random_state
        )
        self._clf.fit(self._encode(texts), y)

    def _predict_proba_fold(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._clf.predict_proba(self._encode(texts)))

    def predict_oof(self, texts, y, n_splits: int = 5) -> np.ndarray:
        """Encode once and run CV on embeddings.

        The encoder is frozen, so precomputing is leak-free and avoids
        re-encoding the corpus once per fold.
        """
        from sklearn.model_selection import StratifiedKFold

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        counts = np.bincount(y, minlength=len(self.classes_))
        folds = int(max(2, min(n_splits, int(counts.min()))))
        if folds < 2:
            return self.predict(texts)

        X = self._encode(texts)
        oof = np.zeros(len(y), dtype=self.classes_.dtype)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        from sklearn.linear_model import LogisticRegression

        for tr_idx, val_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=self.C, solver="lbfgs",
                                     random_state=self.random_state)
            clf.fit(X[tr_idx], y[tr_idx])
            oof[val_idx] = clf.classes_[np.argmax(clf.predict_proba(X[val_idx]), axis=1)]
        self._fold_classes = self.classes_
        return oof


# ----------------------------------------------------------------------------- #
# registry
# ----------------------------------------------------------------------------- #
REGISTRY: dict[str, type[BaseCategorizer]] = {
    "catboost": CatBoostCategorizer,
    "lightgbm": LightGBMCategorizer,
    "xgboost": XGBoostCategorizer,
    "sbert": SentenceTransformerCategorizer,
}

# Measured on the same 2,275-row split: LightGBM fit in 10.0s at 0.9934
# accuracy, CatBoost 131.3s at 0.9802. CatBoost remains available for batch
# runs where its native text handling is worth the wall-clock cost.
DEFAULT_BACKEND = "lightgbm"


def build_categorizer(name: str = DEFAULT_BACKEND, **kwargs) -> BaseCategorizer:
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown categorizer '{name}'. Available: {sorted(REGISTRY)}"
        ) from None
    if name == "sbert" and not SentenceTransformerCategorizer.available():
        raise RuntimeError(
            "The 'sbert' backend needs sentence-transformers and torch, which "
            "are not installed in this environment."
        )
    return cls(**kwargs)


def available_backends() -> list[str]:
    out = [n for n in REGISTRY if n != "sbert"]
    if SentenceTransformerCategorizer.available():
        out.append("sbert")
    return out

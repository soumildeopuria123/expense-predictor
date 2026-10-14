"""Unsupervised anomaly detection built on PyOD.

Why PyOD rather than wiring up a single ``IsolationForest`` by hand: it gives a
uniform ``fit``/``decision_function`` surface over many detectors, which lets us
run an **ensemble**. On financial data a single unsupervised detector produces a
stream of false positives -- "unusual" is not the same as "fraudulent" -- and
requiring several detectors to *agree* before flagging is the standard cure.

Two rules are baked in because getting them wrong silently ruins the model:

* Features are standardised first (``RobustScaler``). ECOD/COPOD/IsolationForest
  are effectively scale-invariant, but LOF and KNN are distance-based and would
  otherwise be dominated by ``spend_prev_7d`` simply because it has the largest
  magnitude.
* Normalisation and thresholds are fitted on the **training** scores and reused
  at scoring time. Renormalising per-call would make scores non-comparable
  between the dashboard and the training report.

Scores are returned in ``[0, 1]`` with **higher meaning more anomalous**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

RANDOM_STATE = 42
ENSEMBLE_MEMBERS = ("ecod", "copod", "iforest")
SINGLETON_METHODS = ("iforest", "ecod", "copod", "lof", "knn")
ALL_METHODS = tuple(sorted(set(SINGLETON_METHODS) | {"ensemble"}))


def _build(name: str, contamination: float, random_state: int):
    if name == "iforest":
        from pyod.models.iforest import IForest

        return IForest(contamination=contamination, random_state=random_state, n_jobs=1)
    if name == "ecod":
        from pyod.models.ecod import ECOD

        return ECOD(contamination=contamination, n_jobs=1)
    if name == "copod":
        from pyod.models.copod import COPOD

        return COPOD(contamination=contamination, n_jobs=1)
    if name == "lof":
        from pyod.models.lof import LOF

        return LOF(n_neighbors=20, contamination=contamination, n_jobs=1)
    if name == "knn":
        from pyod.models.knn import KNN

        return KNN(n_neighbors=10, contamination=contamination, n_jobs=1)
    raise ValueError(f"Unknown detector '{name}'. Choose from {ALL_METHODS}.")


@dataclass(slots=True)
class AnomalyDetector:
    """Fit a detector (or an ensemble) and emit comparable anomaly scores."""

    method: str = "ensemble"
    contamination: float = 0.02
    min_votes: int | None = None          # ensemble only; defaults to a majority
    use_scaler: bool = True
    random_state: int = RANDOM_STATE

    _members: dict = field(default_factory=dict, repr=False)
    _names: tuple[str, ...] = field(default_factory=tuple, repr=False)
    _lo: dict = field(default_factory=dict, repr=False)
    _hi: dict = field(default_factory=dict, repr=False)
    _threshold: dict = field(default_factory=dict, repr=False)
    _scaler: object | None = field(default=None, repr=False)
    fitted_: bool = field(default=False, repr=False)

    # -- helpers -------------------------------------------------------------
    @property
    def member_names(self) -> tuple[str, ...]:
        return self._names

    def _member_specs(self) -> tuple[str, ...]:
        if self.method == "ensemble":
            return ENSEMBLE_MEMBERS
        if self.method not in SINGLETON_METHODS:
            raise ValueError(
                f"Unknown method '{self.method}'. Choose from {ALL_METHODS}."
            )
        return (self.method,)

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self.use_scaler:
            return X
        if fit:
            from sklearn.preprocessing import RobustScaler

            self._scaler = RobustScaler().fit(X)
        if self._scaler is None:
            raise RuntimeError("AnomalyDetector.transform called before fit")
        return self._scaler.transform(X)

    # -- training ------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        Xs = self._scale(X, fit=True)
        self._names = self._member_specs()

        for name in self._names:
            det = _build(name, self.contamination, self.random_state)
            det.fit(Xs)
            self._members[name] = det

            train = np.asarray(det.decision_scores_, dtype=np.float64)
            lo, hi = float(train.min()), float(train.max())
            # guard against a constant score vector
            if hi - lo < 1e-12:
                hi = lo + 1e-12
            self._lo[name], self._hi[name] = lo, hi
            norm = (train - lo) / (hi - lo)
            # Flag rate is set from the training distribution, so contamination
            # means the same thing at score time as it did at fit time.
            self._threshold[name] = float(np.quantile(norm, 1.0 - self.contamination))

        if self.min_votes is None:
            self.min_votes = max(1, (len(self._names) + 1) // 2 + (1 if len(self._names) % 2 == 0 else 0)) \
                if len(self._names) > 1 else 1
            # simple strict majority for odd counts, ceil(n/2)+? -> use ceil
            self.min_votes = int(np.ceil(len(self._names) / 2)) if len(self._names) > 1 else 1

        self.fitted_ = True
        return self

    # -- scoring -------------------------------------------------------------
    def member_scores(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Per-detector scores, each normalised with **training** statistics."""
        if not self.fitted_:
            raise RuntimeError("AnomalyDetector.score called before fit.")
        Xs = self._scale(X, fit=False)
        out: dict[str, np.ndarray] = {}
        for name, det in self._members.items():
            raw = np.asarray(det.decision_function(Xs), dtype=np.float64)
            norm = (raw - self._lo[name]) / (self._hi[name] - self._lo[name])
            out[name] = np.clip(norm, 0.0, 1.0)
        return out

    def score(self, X: np.ndarray) -> np.ndarray:
        """Mean normalised score across members, in [0, 1], higher = worse."""
        scores = self.member_scores(X)
        return np.mean(np.vstack([scores[n] for n in self._names]), axis=0)

    def votes(self, X: np.ndarray) -> np.ndarray:
        """How many members independently consider each row anomalous."""
        scores = self.member_scores(X)
        flags = np.vstack(
            [scores[n] > self._threshold[n] for n in self._names]
        )
        return flags.sum(axis=0).astype(np.int64)

    def labels(self, X: np.ndarray) -> np.ndarray:
        """Boolean flag: True means the row is considered anomalous."""
        if len(self._names) == 1:
            return self.score(X) > self._threshold[self._names[0]]
        assert self.min_votes is not None
        return self.votes(X) >= self.min_votes

    def summary(self) -> dict:
        return {
            "method": self.method,
            "members": list(self._names),
            "contamination": self.contamination,
            "min_votes": self.min_votes,
            "thresholds": {k: round(v, 4) for k, v in self._threshold.items()},
            "fitted": self.fitted_,
        }

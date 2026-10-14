import numpy as np
import pytest

from expense_ml.anomaly import ALL_METHODS, AnomalyDetector
from expense_ml.features import (
    ANOMALY_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FeatureBuilder,
)
from expense_ml.synthetic import ANOMALY_LABEL, make_transactions


@pytest.fixture(scope="module")
def scored():
    df = make_transactions(4000, seed=9)
    cut = int(df.height * 0.7)
    train, test = df.head(cut), df.tail(df.height - cut)
    fb = FeatureBuilder().fit(train)
    cols = [FEATURE_COLUMNS.index(c) for c in ANOMALY_FEATURE_COLUMNS]
    Xtr = fb.fit_transform(train)[:, cols]
    Xte = fb.transform(test, use_history=True)[:, cols]
    return Xtr, Xte, test[ANOMALY_LABEL].to_numpy()


def test_detector_scores_are_comparable_and_bounded(scored):
    Xtr, Xte, y = scored
    d = AnomalyDetector(method="iforest", contamination=0.03).fit(Xtr)
    s = d.score(Xte)
    assert s.shape == (Xte.shape[0],)
    assert np.all((s >= 0) & (s <= 1)), "scores must be normalised to [0,1]"
    assert np.all((s >= 0) & (s <= 1))
    # a score fitted on train must remain in range on unseen rows
    assert d.labels(Xte).dtype == bool


def test_detects_pure_outliers():
    """A sanity check that the detector works at all on obvious outliers."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (400, 3)), np.array([[9.0, 9.0, 9.0]])])
    d = AnomalyDetector(method="iforest", contamination=0.02).fit(X[:400])
    assert d.labels(np.array([[9.0, 9.0, 9.0]]))[0], "obvious outlier not flagged"
    normal = d.labels(rng.normal(0, 1, (50, 3)))
    assert normal.mean() < 0.25, "flagging far too many normal points"


def test_beats_random_on_injected_ground_truth(scored):
    """The whole point: better than guessing, on held-out rows."""
    Xtr, Xte, y = scored
    d = AnomalyDetector(method="iforest", contamination=0.03).fit(Xtr)
    from sklearn.metrics import average_precision_score, f1_score

    ap = average_precision_score(y, d.score(Xte))
    assert ap > y.mean() * 2.0, f"AP {ap:.3f} is not meaningfully above base {y.mean():.3f}"
    assert f1_score(y, d.labels(Xte), zero_division=0) > 0.15


@pytest.mark.parametrize("method", ALL_METHODS)
def test_every_registered_method_fits(scored, method):
    Xtr, Xte, _ = scored
    d = AnomalyDetector(method=method, contamination=0.03).fit(Xtr[:600])
    s = d.score(Xte[:200])
    assert np.isfinite(s).all() and s.shape == (200,)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        AnomalyDetector(method="magic").fit(np.random.rand(50, 3))


def test_scoring_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        AnomalyDetector().score(np.random.rand(10, 3))

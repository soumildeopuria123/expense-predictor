import numpy as np
import pytest

from expense_ml.categorize import (
    REGISTRY,
    available_backends,
    build_categorizer,
)
from expense_ml.synthetic import make_transactions


@pytest.fixture(scope="module")
def corpus():
    df = make_transactions(700, seed=21)
    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform(df["category"].to_numpy())
    return list(df["description"]), y


def test_registry_and_unknown_backend():
    assert "catboost" in REGISTRY and "lightgbm" in REGISTRY
    with pytest.raises(ValueError, match="Unknown categorizer"):
        build_categorizer("does-not-exist")

    backends = available_backends()
    for required in ("catboost", "lightgbm", "xgboost"):
        assert required in backends

    # sbert is an optional extra (pip install -e ".[sbert]"). available_backends()
    # must report it exactly when its dependency is importable -- no more, no
    # less -- so this holds on a bare install *and* with the extra present.
    from expense_ml.categorize import SentenceTransformerCategorizer
    assert ("sbert" in backends) is SentenceTransformerCategorizer.available()


def test_listing_backends_does_not_import_the_optional_stack():
    """Reporting an optional backend must not import it.

    ``available()`` used to do a real ``import sentence_transformers``, which
    adds ~4,000 modules (torch + the whole transformers tree) to the process.
    Streamlit's hot-reload watcher walks every loaded package, and the image
    modules in transformers want ``torchvision`` (not installed) -- so merely
    *opening the dashboard* logged 100+ ``ModuleNotFoundError`` tracebacks
    and made the app look broken.

    Run in a subprocess so the check always starts from a clean interpreter:
    in-process it would silently pass whenever an earlier test had already
    imported torch.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from expense_ml.categorize import available_backends\n"
        "available_backends()\n"
        "heavy = ('sentence_transformers', 'transformers', 'torch')\n"
        "print(','.join(m for m in heavy if m in sys.modules))\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    imported = res.stdout.strip()
    assert not imported, f"available_backends() imported {imported}"

    # And the property the caller actually cares about still holds.
    assert "lightgbm" in available_backends()


def test_fit_predict_shapes(corpus):
    texts, y = corpus
    n_classes = len(np.unique(y))
    m = build_categorizer("lightgbm", n_estimators=40).fit(texts, y)
    proba = m.predict_proba(texts)
    assert proba.shape == (len(texts), n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(m.predict(texts)) <= set(m.classes_)


def test_predict_before_fit_raises(corpus):
    with pytest.raises(RuntimeError, match="before fit"):
        build_categorizer("lightgbm").predict_proba(["COFFEE"])


def test_oof_predictions_are_labelled_and_per_row(corpus):
    texts, y = corpus
    m = build_categorizer("lightgbm", n_estimators=40)
    oof = m.predict_oof(texts, y, n_splits=3)
    assert oof.shape == y.shape
    assert set(np.unique(oof)) <= set(np.unique(y)), "OOF produced unknown labels"
    assert isinstance(oof, np.ndarray)


def test_oof_agrees_with_independent_holdout(corpus):
    """OOF must land near a separately-computed holdout accuracy.

    This is the real correctness property: if the cross-validation plumbing
    were broken (misaligned indices, in-sample predictions leaking through),
    OOF would diverge sharply from a plain train/test split on the same data.
    An absolute accuracy target would be meaningless here -- with ~470
    training rows across 17 classes, LightGBM+TF-IDF scores ~0.55, whereas a
    linear model would score ~0.89. That gap is a property of the booster on
    sparse text, not of the CV.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    texts, y = corpus
    rng = np.random.RandomState(0)

    tr, te = train_test_split(
        np.arange(len(y)), test_size=0.33, random_state=0, stratify=y
    )
    m = build_categorizer("lightgbm", n_estimators=120)
    m.fit([texts[i] for i in tr], y[tr])
    holdout = accuracy_score(y[te], m.predict([texts[i] for i in te]))

    oof_model = build_categorizer("lightgbm", n_estimators=120)
    oof = accuracy_score(y, oof_model.predict_oof(texts, y, n_splits=3))

    majority = np.bincount(y).max() / len(y)
    assert oof > majority * 2, f"OOF {oof:.3f} barely above majority-class {majority:.3f}"
    assert abs(oof - holdout) < 0.15, (
        f"OOF {oof:.3f} disagrees with independent holdout {holdout:.3f} by more "
        "than 0.15 -- the cross-validation plumbing is suspect"
    )


def test_tiny_corpus_does_not_crash_catboost():
    """CatBoost's default dictionaries raise 'Dictionary size is 0' on small
    data; our fallback path must handle it."""
    pytest.importorskip("catboost")
    texts = ["CHIPOTLE #1", "WHOLE FOODS", "UBER TRIP", "STARBUCKS",
             "PAYROLL DEP", "RENT PAYMENT", "NETFLIX", "SHELL OIL"]
    y = np.array([1, 0, 2, 1, 4, 3, 6, 2])
    m = build_categorizer("catboost", iterations=25).fit(texts, y)
    assert m.predict_proba(texts).shape == (8, len(np.unique(y)))

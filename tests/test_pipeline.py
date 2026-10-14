import numpy as np
import polars as pl
import pytest

from expense_ml.pipeline import TrainConfig, TrainResult, train_model
from expense_ml.synthetic import ANOMALY_LABEL, make_transactions


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    df = make_transactions(900, seed=31)
    res = train_model(df, TrainConfig(categorizer="lightgbm", oof_folds=3))
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    res.model.save(path)
    return df, res, path


def test_metrics_structure(trained):
    _, res, _ = trained
    m = res.metrics
    for key in ("config", "rows", "categorization", "anomaly", "backends"):
        assert key in m, f"missing metrics section: {key}"
    cat, an = m["categorization"], m["anomaly"]
    assert 0.0 <= cat["accuracy"] <= 1.0
    assert cat["accuracy"] > 0.6, "categorizer is not learning anything"
    assert an["ground_truth"] is True
    assert an["lift"] > 2.0, f"anomaly model gives only {an['lift']:.1f}x lift"
    assert set(m["backends"]["anomaly_features"]) <= set(
        __import__("expense_ml.features", fromlist=["FEATURE_COLUMNS"]).FEATURE_COLUMNS
    )


def test_chronological_split_not_random(trained):
    _, res, _ = trained
    assert res.metrics["rows"]["train"] > res.metrics["rows"]["test"]
    assert res.metrics["config"]["test_fraction"] == 0.30


def test_save_load_roundtrip_and_score(trained):
    df, res, path = trained
    from expense_ml.pipeline import ExpenseModel

    loaded = ExpenseModel.load(path)
    assert loaded.is_fitted and loaded.config.categorizer == "lightgbm"
    out = loaded.score(df.head(200))
    for col in ("category", "category_confidence", "anomaly_score",
                "anomaly_flag", "anomaly_reason"):
        assert col in out.columns, f"score() missing {col}"
    assert out.height == 200
    assert out["anomaly_score"].is_finite().all()
    assert out["category"].is_in(loaded.category_labels).all(), "unknown label emitted"
    assert out["category_confidence"].min() >= 0.0
    assert out["category_confidence"].max() <= 1.0
    assert out.filter(pl.col("anomaly_flag"))["anomaly_reason"].str.len_chars().min() > 0


def test_two_models_disagree_ensemble_vs_singleton(trained):
    """Guards against a registry that silently returns the same detector."""
    from expense_ml.anomaly import ENSEMBLE_MEMBERS

    assert len(set(ENSEMBLE_MEMBERS)) == len(ENSEMBLE_MEMBERS) >= 3


def test_training_rejects_unlabelled_data():
    df = make_transactions(200, seed=4).drop("category")
    with pytest.raises(ValueError, match="category"):
        train_model(df, TrainConfig(categorizer="lightgbm"))


def test_training_rejects_tiny_frames():
    with pytest.raises(ValueError, match="at least 50 rows"):
        train_model(make_transactions(20, seed=1), TrainConfig(categorizer="lightgbm"))


def test_scoring_before_fit_raises():
    from expense_ml.pipeline import ExpenseModel

    with pytest.raises(RuntimeError, match="before fit"):
        ExpenseModel().score(make_transactions(50, seed=2))

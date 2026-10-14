import datetime as dt

import numpy as np
import polars as pl

from expense_ml.config import ACCOUNT, AMOUNT, CATEGORY, DATE, DESC
from expense_ml.features import ANOMALY_FEATURE_COLUMNS, FEATURE_COLUMNS, FeatureBuilder
from expense_ml.synthetic import make_transactions


def test_matrix_shape_and_finiteness():
    df = make_transactions(800, seed=3)
    X = FeatureBuilder().fit_transform(df)
    assert X.shape == (df.height, len(FEATURE_COLUMNS))
    assert np.isfinite(X).all(), "NaN/inf would poison every downstream model"


def test_anomaly_subset_exists():
    missing = set(ANOMALY_FEATURE_COLUMNS) - set(FEATURE_COLUMNS)
    assert not missing, f"unknown columns in anomaly subset: {missing}"


def test_no_future_leakage():
    """Appending a wild future row must not change earlier rows' features.

    This is the regression guard for the rolling-window implementation: a
    naive 'rolling over the whole frame' would let row 900 see row 901.
    """
    df = make_transactions(900, seed=5)
    train, test = df.head(650), df.tail(df.height - 650)
    fb = FeatureBuilder().fit(train)
    before = fb.transform(test)

    wild = test.head(1).with_columns(
        pl.lit(dt.datetime(2035, 1, 1)).alias(DATE),
        pl.lit(-999_999.0).alias(AMOUNT),
    )
    after = fb.transform(pl.concat([test, wild], how="vertical"))[: test.height]

    np.testing.assert_allclose(before, after, err_msg="future rows leaked into features")


def test_past_rows_inside_window_are_respected():
    """Negative control: history must actually be used, not just excluded."""
    df = make_transactions(900, seed=5)
    train, test = df.head(650), df.tail(df.height - 650)
    fb = FeatureBuilder().fit(train)
    before = fb.transform(test)

    first_t = test[DATE].min()
    first_acct = test.filter(pl.col(DATE) == first_t)[ACCOUNT][0]
    extra = test.head(1).with_columns(
        pl.lit(first_t - dt.timedelta(days=1)).alias(DATE),
        pl.lit(-500.0).alias(AMOUNT),
        pl.lit(first_acct).alias(ACCOUNT),
    )
    after = fb.transform(pl.concat([test, extra], how="vertical"))[: test.height]

    i7 = FEATURE_COLUMNS.index("txns_prev_7d")
    ispend = FEATURE_COLUMNS.index("spend_prev_7d")
    assert (after[:, i7] - before[:, i7]).max() >= 1.0, "7d count ignored new history"
    assert (after[:, ispend] - before[:, ispend]).max() >= 500.0, "7d spend ignored new history"


def test_burst_window_sees_a_burst():
    """A tight cluster of transactions must move the 2h counter."""
    base = make_transactions(400, seed=2).head(300)
    anchor = base["date"].max()
    burst = []
    for k in range(6):
        burst.append(base.head(1).with_columns(
            pl.lit(anchor + dt.timedelta(minutes=k * 3)).alias(DATE),
            pl.lit(-49.99).alias(AMOUNT),
        ))
    frame = pl.concat([base] + burst, how="vertical").sort(DATE)
    X = FeatureBuilder().fit(frame).transform(frame)
    i2h = FEATURE_COLUMNS.index("txns_prev_2h")
    assert X[:, i2h].max() >= 5, "2h window failed to detect the injected burst"

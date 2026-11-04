import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from credit_risk import config
from credit_risk.cleaning import clean
from credit_risk.preprocessing import feature_columns, make_feature_pipeline


@pytest.fixture
def clean_big(raw_df) -> pd.DataFrame:
    df = clean(pd.concat([raw_df] * 60, ignore_index=True))
    df["id"] = np.arange(len(df))
    return df


def test_feature_columns_partition():
    num, cat = feature_columns(status_as_category=True)
    assert not set(num) & set(cat)
    assert set(config.PAY_STATUS) <= set(cat)
    assert not set(config.PAY_STATUS) & set(num)


def test_output_is_pandas_with_stable_names(clean_big):
    out = make_feature_pipeline().fit_transform(clean_big)
    assert isinstance(out, pd.DataFrame)
    assert "util_mean" in out and "education_1" in out
    assert "id" not in out and config.TARGET not in out


def test_scaled_numeric_columns_have_zero_mean(clean_big):
    out = make_feature_pipeline(scaled=True).fit_transform(clean_big)
    assert np.allclose(out["limit_bal"].mean(), 0, atol=1e-9)


def test_unscaled_keeps_original_values(clean_big):
    out = make_feature_pipeline(scaled=False).fit_transform(clean_big)
    assert out["limit_bal"].tolist() == clean_big["limit_bal"].tolist()


def test_unseen_category_does_not_break(clean_big):
    pipe = make_feature_pipeline().fit(clean_big)
    new = clean_big.head(3).copy()
    new["education"] = 99
    out = pipe.transform(new)
    assert out.filter(like="education_").to_numpy().sum(axis=1).max() <= 1
    assert np.isfinite(out.to_numpy()).all()


def test_rare_status_values_become_infrequent(clean_big):
    clean_big.loc[:2, config.PAY_STATUS] = 7
    out = make_feature_pipeline(status_as_category=True).fit_transform(clean_big)
    assert not any(c.endswith("_7") for c in out.columns if c.startswith("pay_status"))


def test_statistics_come_only_from_fit_data(clean_big):
    train, other = clean_big.iloc[:120], clean_big.iloc[120:].copy()
    other["limit_bal"] *= 100
    pipe = make_feature_pipeline().fit(train)
    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    idx = list(scaler.feature_names_in_).index("limit_bal")
    assert scaler.mean_[idx] == pytest.approx(train["limit_bal"].mean())


def test_clone_drops_fitted_state(clean_big):
    fitted = make_feature_pipeline().fit(clean_big)
    copy = clone(fitted)
    assert not hasattr(copy.named_steps["prep"], "transformers_")
    pd.testing.assert_frame_equal(copy.fit_transform(clean_big), fitted.transform(clean_big))

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from credit_risk import config
from credit_risk.balancing import STRATEGIES, BalancedWeights, balanced_pipeline
from credit_risk.cleaning import clean


@pytest.fixture
def imbalanced():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=2_000)})
    y = pd.Series((X["a"] + rng.normal(size=2_000) > 1.5).astype(int))
    return X, y


def test_balanced_weights_inflate_mean_probability(imbalanced):
    X, y = imbalanced
    plain = LogisticRegression().fit(X, y).predict_proba(X)[:, 1].mean()
    weighted = BalancedWeights(LogisticRegression()).fit(X, y).predict_proba(X)[:, 1].mean()
    assert plain == pytest.approx(y.mean(), abs=0.01)
    assert weighted > plain + 0.15


def test_balanced_weights_is_clonable(imbalanced):
    X, y = imbalanced
    model = clone(BalancedWeights(LogisticRegression())).fit(X, y)
    assert list(model.classes_) == [0, 1]
    assert model.predict(X).shape == (len(X),)


@pytest.fixture
def clean_big(raw_df) -> pd.DataFrame:
    df = clean(pd.concat([raw_df] * 60, ignore_index=True))
    df.loc[df.index % 4 == 0, config.TARGET] = 1
    df.loc[df.index % 4 != 0, config.TARGET] = 0
    return df


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_fits(clean_big, strategy):
    pipe = balanced_pipeline("logistic", strategy).fit(clean_big, clean_big[config.TARGET])
    proba = pipe.predict_proba(clean_big)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_smote_does_not_resample_at_predict_time(clean_big):
    pipe = balanced_pipeline("logistic", "smote").fit(clean_big, clean_big[config.TARGET])
    assert len(pipe.predict_proba(clean_big)) == len(clean_big)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        balanced_pipeline("logistic", "oversample")

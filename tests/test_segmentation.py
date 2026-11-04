import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from credit_risk import config
from credit_risk.segmentation import (
    choose_n_components,
    explained_variance_table,
    loadings,
    make_pca_pipeline,
)


@pytest.fixture
def clean_sample() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    level = rng.random(n)
    df = pd.DataFrame({"limit_bal": rng.integers(10, 100, n) * 1_000})
    for i, (b, p, s) in enumerate(
        zip(config.BILL_AMT, config.PAY_AMT, config.PAY_STATUS, strict=True)
    ):
        df[b] = (level * df["limit_bal"] * (1 + 0.05 * i)).round()
        df[p] = (rng.random(n) * 3_000).round()
        df[s] = rng.integers(-2, 3, n)
    return df


@pytest.mark.parametrize(
    ("cumulative", "expected"),
    [([0.5, 0.79, 0.85, 1.0], 3), ([0.8, 0.9, 1.0], 1), ([0.2, 0.5, 1.0], 3)],
)
def test_choose_n_components(cumulative, expected):
    assert choose_n_components(pd.Series(cumulative)) == expected


def test_pca_pipeline_variance_table(clean_sample):
    pipe = make_pca_pipeline().fit(clean_sample)
    table = explained_variance_table(pipe.named_steps["pca"])
    assert len(table) == len(config.BEHAVIOR_FEATURES)
    assert table["acumulada"].iloc[-1] == pytest.approx(1.0)
    assert table["variancia"].is_monotonic_decreasing


def test_loadings_are_unit_vectors(clean_sample):
    pipe = make_pca_pipeline(3).fit(clean_sample)
    load = loadings(pipe.named_steps["pca"], config.BEHAVIOR_FEATURES)
    assert load.shape == (17, 3)
    assert np.allclose((load**2).sum(axis=0), 1.0)


def test_pca_pipeline_is_clonable_and_reproducible(clean_sample):
    a = make_pca_pipeline(3).fit_transform(clean_sample)
    b = clone(make_pca_pipeline(3)).fit_transform(clean_sample)
    assert np.allclose(a, b)

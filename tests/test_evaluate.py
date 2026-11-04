import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from credit_risk.evaluate import cv_scores, make_cv, paired_comparison


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=500) > 0.8).astype(int))
    return X, y


def test_cv_has_15_folds():
    assert make_cv().get_n_splits() == 15


def test_cv_scores_shape_and_brier_positive(xy):
    X, y = xy
    out = cv_scores(LogisticRegression(), X, y)
    assert out.shape == (15, 3)
    assert (out["brier"] > 0).all()


def test_same_seed_gives_same_folds(xy):
    X, y = xy
    pd.testing.assert_frame_equal(
        cv_scores(LogisticRegression(), X, y), cv_scores(LogisticRegression(), X, y)
    )


def test_paired_comparison_detects_better_model(xy):
    X, y = xy
    base = cv_scores(DummyClassifier(strategy="prior"), X, y)
    cand = cv_scores(LogisticRegression(), X, y)
    res = paired_comparison(base, cand, "pr_auc")
    assert res["ganho_medio"] > 0
    assert res["win_rate"] == 1.0


def test_paired_comparison_brier_lower_is_better():
    base = pd.DataFrame({"brier": [0.20, 0.20]})
    cand = pd.DataFrame({"brier": [0.10, 0.30]})
    res = paired_comparison(base, cand, "brier")
    assert res["win_rate"] == 0.5
    assert res["ganho_medio"] == pytest.approx(0.0)

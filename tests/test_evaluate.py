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
    assert out.shape == (15, 5)
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


def test_ks_perfect_and_random():
    from credit_risk.evaluate import ks_statistic

    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_ks_known_value():
    from credit_risk.evaluate import ks_statistic

    y = np.array([0, 0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.6, 0.9])
    assert ks_statistic(y, score) == pytest.approx(0.75)


def test_summarize_orders_by_metric_and_compares_to_reference(xy):
    from credit_risk.evaluate import summarize

    X, y = xy
    scores = {
        "dummy": cv_scores(DummyClassifier(strategy="prior"), X, y),
        "logit": cv_scores(LogisticRegression(), X, y),
    }
    table = summarize(scores, reference="dummy")
    assert table.loc[0, "modelo"] == "logit"
    assert table.loc[0, "win_rate_vs_dummy"] == 1.0
    assert {"ks_media", "accuracy_media"} <= set(table.columns)


def test_reliability_table_on_perfectly_calibrated_probabilities():
    from credit_risk.evaluate import reliability_table

    rng = np.random.default_rng(3)
    p = rng.random(50_000)
    y = (rng.random(50_000) < p).astype(int)
    table = reliability_table(y, p)
    assert len(table) == 10 and table["n"].sum() == 50_000
    assert np.allclose(table["prevista"], table["observada"], atol=0.02)


def test_oof_predictions_cover_every_row(xy):
    from credit_risk.evaluate import oof_predictions

    X, y = xy
    p = oof_predictions(LogisticRegression(), X, y)
    assert p.shape == (len(X),) and ((p >= 0) & (p <= 1)).all()

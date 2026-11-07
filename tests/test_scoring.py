import numpy as np
import pandas as pd
import pytest

from credit_risk import config
from credit_risk.scoring import (
    ScoringParams,
    assign_band,
    fit_scoring_params,
    pd_to_score,
    score_from_pd,
    score_to_pd,
)


@pytest.mark.parametrize(
    ("pd_", "score"), [(0.2212, 500), (0.1243, 560), (0.0662, 620), (0.3618, 440)]
)
def test_worked_examples(pd_, score):
    assert pd_to_score(pd_) == pytest.approx(score, abs=1)


def test_score_is_monotone_decreasing_in_pd_and_bounded():
    s = pd_to_score(np.linspace(0, 1, 101))
    assert (np.diff(s) <= 0).all()
    assert s.min() >= 0 and s.max() <= 1000


def test_pdo_doubles_the_odds():
    p1, p2 = score_to_pd(500), score_to_pd(560)
    assert ((1 - p2) / p2) / ((1 - p1) / p1) == pytest.approx(2.0)


def test_score_to_pd_inverts_pd_to_score():
    p = np.array([0.05, 0.2, 0.5, 0.8])
    assert score_to_pd(pd_to_score(p)) == pytest.approx(p, abs=0.005)


def test_band_shares_respected_on_fitting_data():
    rng = np.random.default_rng(0)
    pd_ = rng.beta(2, 7, 20_000)
    ead = rng.uniform(0, 50_000, 20_000)
    params = fit_scoring_params(pd_, ead)
    bands = pd.Series(assign_band(pd_to_score(pd_), params.band_cuts)).value_counts(normalize=True)
    for band, share in config.BAND_SHARES.items():
        assert bands[band] == pytest.approx(share, abs=0.02)


def test_assign_band_orders_a_to_e():
    cuts = {"A": 600, "B": 550, "C": 500, "D": 450}
    assert list(assign_band(np.array([700, 560, 500, 449, 450]), cuts)) == ["A", "B", "C", "E", "D"]


def test_score_from_pd_chain():
    params = ScoringParams(band_cuts={"A": 600, "B": 550, "C": 500, "D": 450}, ev_cut=0.0)
    out = score_from_pd(np.array([0.2]), np.array([10_000]), params)
    row = out.iloc[0]
    assert row["perda_esperada"] == pytest.approx(0.2 * config.LGD * 10_000)
    assert row["faixa"] in config.BAND_SHARES
    assert row["acao"] == config.BAND_ACTIONS[row["faixa"]]
    assert bool(row["prioritario"]) == bool(row["valor_esperado_acao"] >= 0)

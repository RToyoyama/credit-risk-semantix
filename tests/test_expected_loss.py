import numpy as np
import pytest

from credit_risk.expected_loss import (
    concentration_curve,
    expected_loss,
    ranking_comparison,
    top_mask,
)


def test_expected_loss_worked_example():
    assert expected_loss(np.array([0.20]), np.array([10_000]), lgd=0.6)[0] == pytest.approx(1_200)


def test_top_mask_selects_largest():
    assert np.flatnonzero(top_mask(np.array([1, 5, 3, 4, 2, 0, 9, 8, 7, 6]), 0.2)).tolist() == [
        6,
        7,
    ]


def test_ranking_comparison_prefers_large_exposure():
    pd_ = np.array([0.60, 0.25] + [0.05] * 18)
    ead = np.array([2_000, 50_000] + [1_000] * 18)
    y = np.array([1, 1] + [0] * 18)
    out = ranking_comparison(pd_, ead, y, share=0.05, lgd=0.6)
    assert out["topo"] == 1
    assert out["mudam_de_prioridade"] == 1
    assert out["perda_real_capturada_por_el"] > out["perda_real_capturada_por_pd"]


def test_concentration_curve_ends_at_one_and_is_monotone():
    rng = np.random.default_rng(0)
    v = rng.random(1_000)
    curve = concentration_curve(v, v, points=20)
    assert curve["fracao_capturada"].iloc[-1] == pytest.approx(1.0)
    assert curve["fracao_capturada"].is_monotonic_increasing

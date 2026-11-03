import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from credit_risk.stats import (
    bootstrap_ci,
    chi2_cramers_v,
    effect_magnitude,
    holm_adjust,
    mann_whitney_rank_biserial,
    rate_by_group,
    results_table,
)


@pytest.fixture
def y() -> pd.Series:
    return pd.Series([1] * 50 + [0] * 50, name="default")


def test_bootstrap_ci_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    x = (rng.random(2_000) < 0.22).astype(int)
    est, low, high = bootstrap_ci(x, n_boot=2_000)
    assert low < est < high
    assert est == pytest.approx(x.mean())
    assert bootstrap_ci(x, n_boot=2_000) == (est, low, high)


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(2)
    small = rng.random(200) < 0.22
    large = rng.random(20_000) < 0.22
    _, l1, h1 = bootstrap_ci(small, n_boot=1_000)
    _, l2, h2 = bootstrap_ci(large, n_boot=1_000)
    assert (h2 - l2) < (h1 - l1)


def test_cramers_v_perfect_association(y):
    x = y.map({1: "a", 0: "b"}).rename("x")
    assert chi2_cramers_v(x, y).effect == pytest.approx(1.0)


def test_cramers_v_independence(y):
    x = pd.Series(["a", "b"] * 50, name="x")
    res = chi2_cramers_v(x, y)
    assert res.effect == pytest.approx(0.0)
    assert res.p_value == pytest.approx(1.0)


def test_chi2_matches_scipy_without_yates(y):
    x = pd.Series(["a"] * 35 + ["b"] * 15 + ["a"] * 20 + ["b"] * 30, name="x")
    expected = sps.chi2_contingency(pd.crosstab(x, y), correction=False)[0]
    assert chi2_cramers_v(x, y).statistic == pytest.approx(expected)


@pytest.mark.parametrize(("shift", "expected"), [(100, 1.0), (-100, -1.0)])
def test_rank_biserial_extremes(y, shift, expected):
    values = pd.Series(np.where(y == 1, shift, 0) + np.arange(100) % 10, name="v")
    assert mann_whitney_rank_biserial(values, y).effect == pytest.approx(expected)


def test_rank_biserial_zero_for_identical_distributions(y):
    values = pd.Series(np.tile(np.arange(50), 2), name="v")
    assert mann_whitney_rank_biserial(values, y).effect == pytest.approx(0.0)


def test_holm_known_example():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


@pytest.mark.parametrize(
    ("effect", "label"),
    [
        (0.0, "desprezível"),
        (0.099, "desprezível"),
        (-0.10, "pequeno"),
        (0.3, "médio"),
        (0.7, "grande"),
    ],
)
def test_effect_magnitude(effect, label):
    assert effect_magnitude(effect) == label


def test_results_table_requires_significance_and_effect(y):
    strong = y.map({1: "a", 0: "b"}).rename("forte")
    weak = pd.Series(["a", "b"] * 50, name="fraca")
    table = results_table([chi2_cramers_v(weak, y), chi2_cramers_v(strong, y)])
    assert table.loc[0, "variable"] == "forte"
    assert table.set_index("variable")["relevante"].to_dict() == {"forte": True, "fraca": False}


def test_rate_by_group_with_wilson_ci():
    df = pd.DataFrame(
        {"g": ["a"] * 10 + ["b"] * 10, "default": [1] * 2 + [0] * 8 + [1] * 5 + [0] * 5}
    )
    out = rate_by_group(df, "g").set_index("g")
    assert out.loc["a", "taxa"] == pytest.approx(0.2)
    assert out.loc["b", "taxa"] == pytest.approx(0.5)
    assert (out["ic_inf"] < out["taxa"]).all() and (out["taxa"] < out["ic_sup"]).all()

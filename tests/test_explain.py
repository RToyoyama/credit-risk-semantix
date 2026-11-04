import numpy as np
import pandas as pd
import pytest

from credit_risk.explain import odds_ratio_table, vif_table


def test_vif_is_one_for_independent_and_high_for_duplicated():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=2_000), rng.normal(size=2_000)
    X = pd.DataFrame({"a": a, "b": b, "a_copy": a + rng.normal(scale=0.05, size=2_000)})
    vif = vif_table(X)
    assert vif["b"] == pytest.approx(1.0, abs=0.05)
    assert vif["a"] > 100 and vif["a_copy"] > 100


def test_odds_ratio_recovers_known_coefficient():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20_000)
    p = 1 / (1 + np.exp(-(-1.0 + np.log(2.0) * x)))
    y = pd.Series((rng.random(20_000) < p).astype(int))
    table = odds_ratio_table(pd.DataFrame({"x": x}), y)
    assert table.loc["x", "razao_chances"] == pytest.approx(2.0, rel=0.05)
    assert table.loc["x", "ic_inf"] < 2.0 < table.loc["x", "ic_sup"]
    assert 0 < table.attrs["pseudo_r2"] < 1

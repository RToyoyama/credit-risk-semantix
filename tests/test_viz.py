import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from credit_risk.stats import rate_by_group
from credit_risk.viz import pct, plot_rate_by_group, save


@pytest.mark.parametrize(("value", "decimals", "text"), [(0.2212, 1, "22,1%"), (0.5, 0, "50%")])
def test_pct_uses_brazilian_decimal(value, decimals, text):
    assert pct(value, decimals) == text


def test_plot_rate_by_group_draws_one_bar_per_group(tmp_path):
    df = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10, "default": [1, 0] * 10})
    ax = plot_rate_by_group(rate_by_group(df, "g"), "g", "título", overall=0.5, highlight=["a"])
    assert len(ax.patches) == 2
    assert save(ax.figure, "teste", folder=tmp_path).exists()

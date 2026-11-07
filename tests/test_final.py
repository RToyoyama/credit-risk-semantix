import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from credit_risk import config
from credit_risk.final import (
    HoldoutAlreadyUsedError,
    group_report,
    metrics_with_ci,
    partial_dependence,
    run_final,
)


def test_run_final_refuses_to_reuse_holdout(tmp_path, monkeypatch):
    sealed = tmp_path / "holdout_metrics.csv"
    sealed.write_text("métrica,valor\n")
    monkeypatch.setattr(config, "HOLDOUT_METRICS", sealed)
    with pytest.raises(HoldoutAlreadyUsedError):
        run_final()


def test_metrics_with_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    p = rng.random(1_000)
    y = (rng.random(1_000) < p).astype(int)
    out = metrics_with_ci(y, p, n_boot=200).set_index("métrica")
    assert set(out.index) == {"pr_auc", "roc_auc", "ks", "brier"}
    assert (out["ic_inf"] <= out["valor"]).all() and (out["valor"] <= out["ic_sup"]).all()


def test_group_report_calibration_gap():
    y = np.array([1, 0, 0, 0, 1, 1, 0, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.9, 0.1, 0.3])
    act = np.array([True, False, False, False, True, True, False, False])
    out = group_report(y, p, act, pd.Series(list("aaaabbbb"))).set_index("grupo")
    assert out.loc["a", "gap_calibracao"] == pytest.approx(0.5 - 0.25)
    assert out.loc["b", "pct_acionados"] == pytest.approx(0.5)
    assert out.loc["a", "precisao_acionados"] == pytest.approx(1.0)


def test_partial_dependence_increases_with_monotone_feature():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=500)})
    y = (X["x"] + rng.normal(size=500) > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    out = partial_dependence(model, X, "x", [-2, 0, 2])
    assert out["pd_media"].is_monotonic_increasing

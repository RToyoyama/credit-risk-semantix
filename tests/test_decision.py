import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from credit_risk import config
from credit_risk.cleaning import clean
from credit_risk.decision import (
    Premises,
    bootstrap_diff,
    calibrated_pipeline,
    capacity_mask,
    delay_rules,
    expected_value,
    exposure,
    policy_result,
    realized_value,
    rule_priority,
    threshold_curve,
)


@pytest.fixture
def clean_big(raw_df) -> pd.DataFrame:
    df = clean(pd.concat([raw_df] * 60, ignore_index=True))
    df[config.TARGET] = (df.index % 4 == 0).astype(int)
    return df


def test_none_returns_plain_pipeline():
    assert isinstance(calibrated_pipeline("none"), Pipeline)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrated_pipeline_fits_and_predicts(clean_big, method):
    model = calibrated_pipeline(method)
    assert isinstance(model, CalibratedClassifierCV)
    proba = model.fit(clean_big, clean_big[config.TARGET]).predict_proba(clean_big)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        calibrated_pipeline("beta")


PREM = Premises(lgd=0.6, efficacy=0.3, action_cost=100, friction_cost=300)


def test_exposure_clips_negative_bills():
    df = pd.DataFrame({"bill_amt_1": [-500, 0, 10_000]})
    assert exposure(df).tolist() == [0, 0, 10_000]


def test_realized_value_matches_worked_example():
    assert realized_value(np.array([1, 0]), np.array([10_000, 10_000]), PREM).tolist() == [
        1_700,
        -400,
    ]


def test_expected_value_break_even_matches_formula():
    ead = 10_000.0
    break_even = 400 / (0.6 * ead * 0.3 + 300)
    assert expected_value(np.array([break_even]), np.array([ead]), PREM)[0] == pytest.approx(0)
    assert break_even == pytest.approx(0.1905, abs=1e-4)


def test_policy_result_counts_and_value():
    y = np.array([1, 0, 1, 0])
    ead = np.full(4, 10_000.0)
    res = policy_result(np.array([True, True, False, False]), y, ead, PREM)
    assert res["acionados"] == 2 and res["calotes_alcancados"] == 1
    assert res["valor"] == pytest.approx(1_700 - 400)
    assert res["perda_sem_acao"] == pytest.approx(2 * 0.6 * 10_000)


def test_no_action_is_zero_and_curve_monotone_in_count():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    ead = np.full(500, 5_000.0)
    assert policy_result(np.zeros(500, bool), y, ead, PREM)["valor"] == 0
    curve = threshold_curve(p, y, ead, PREM, step=0.1)
    assert curve["acionados"].is_monotonic_decreasing


def test_delay_rules_flag_late_clients():
    df = pd.DataFrame({c: [0, 2] for c in config.PAY_STATUS})
    rules = delay_rules(df)
    assert rules["atraso em set >= 2 meses"].tolist() == [False, True]
    assert rules["meses com atraso >= 4"].tolist() == [False, True]


def test_bootstrap_diff_is_paired_and_contains_estimate():
    rng = np.random.default_rng(1)
    a = rng.normal(10, 5, 2_000)
    b = a - 1
    diff, low, high = bootstrap_diff(a, b, n_boot=500)
    assert diff == pytest.approx(2_000)
    assert low == pytest.approx(2_000) and high == pytest.approx(2_000)


def test_capacity_mask_picks_top_scores():
    score = np.array([0.1, 0.9, 0.5, 0.7, 0.3, 0.2, 0.8, 0.4, 0.6, 0.0])
    mask = capacity_mask(score, 0.2)
    assert mask.sum() == 2
    assert set(np.flatnonzero(mask)) == {1, 6}


def test_capacity_mask_respects_eligibility():
    score = np.arange(10, dtype=float)
    eligible = score < 3
    mask = capacity_mask(score, 0.5, eligible)
    assert set(np.flatnonzero(mask)) == {0, 1, 2}


def test_rule_priority_orders_by_recent_delay_then_months_then_bill():
    df = pd.DataFrame(
        {
            "bill_amt_1": [1_000, 9_000, 5_000, 5_000],
            **{c: [0, 0, 0, 0] for c in config.PAY_STATUS},
        }
    )
    df.loc[0, "pay_status_1"] = 2
    df.loc[2, ["pay_status_2", "pay_status_3"]] = 1
    order = np.argsort(-rule_priority(df))
    assert order.tolist() == [0, 2, 1, 3]

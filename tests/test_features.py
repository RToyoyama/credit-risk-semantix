import numpy as np
import pandas as pd
import pytest

from credit_risk import config
from credit_risk.features import (
    behavior_features,
    build_features,
    delay_months,
    engineered_features,
    payment_ratio,
    utilization,
)


@pytest.fixture
def clean_rows() -> pd.DataFrame:
    row = {
        "limit_bal": 10_000,
        **dict.fromkeys(config.PAY_STATUS, 0),
        **dict.fromkeys(config.BILL_AMT, 5_000),
        **dict.fromkeys(config.PAY_AMT, 1_000),
    }
    return pd.DataFrame([row, row])


def test_utilization_is_bill_over_limit(clean_rows):
    assert utilization(clean_rows)["utilization_1"].tolist() == [0.5, 0.5]


@pytest.mark.parametrize(("bill", "expected"), [(-3_000, 0.0), (50_000, 1.5)])
def test_utilization_is_clipped(clean_rows, bill, expected):
    clean_rows.loc[0, "bill_amt_1"] = bill
    assert utilization(clean_rows).loc[0, "utilization_1"] == expected


def test_payment_ratio_uses_previous_month_bill(clean_rows):
    clean_rows.loc[0, "pay_amt_1"] = 2_500
    clean_rows.loc[0, "bill_amt_2"] = 10_000
    assert payment_ratio(clean_rows).loc[0, "payment_ratio_1"] == 0.25


@pytest.mark.parametrize(("bill", "paid"), [(0, 0), (-500, 100), (1_000, 9_000)])
def test_payment_ratio_is_one_when_nothing_owed_or_overpaid(clean_rows, bill, paid):
    clean_rows.loc[0, "bill_amt_2"] = bill
    clean_rows.loc[0, "pay_amt_1"] = paid
    assert payment_ratio(clean_rows).loc[0, "payment_ratio_1"] == 1.0


def test_payment_ratio_has_five_months(clean_rows):
    assert list(payment_ratio(clean_rows).columns) == config.PAYMENT_RATIO


def test_delay_ignores_non_positive_status(clean_rows):
    clean_rows.loc[0, config.PAY_STATUS] = [-2, -1, 0, 1, 2, 8]
    assert delay_months(clean_rows).loc[0].tolist() == [0, 0, 0, 1, 2, 8]


def test_behavior_features_shape_and_no_target(clean_rows):
    clean_rows[config.TARGET] = 1
    out = behavior_features(clean_rows)
    assert list(out.columns) == config.BEHAVIOR_FEATURES
    assert out.shape == (2, 17)
    assert np.isfinite(out.to_numpy()).all()


def test_features_are_row_wise(clean_rows):
    single = behavior_features(clean_rows.iloc[[0]])
    both = behavior_features(clean_rows)
    pd.testing.assert_frame_equal(single, both.iloc[[0]])


def test_engineered_features_columns_and_finite(clean_rows):
    out = engineered_features(clean_rows)
    assert list(out.columns) == config.ENGINEERED_FEATURES
    assert np.isfinite(out.to_numpy(dtype=float)).all()


def test_delay_summaries(clean_rows):
    clean_rows.loc[0, config.PAY_STATUS] = [3, 2, 0, -1, -2, 0]
    out = engineered_features(clean_rows).loc[0]
    assert out["delay_max"] == 3
    assert out["delay_months"] == 2
    assert out["delay_recent"] == 3
    assert out["delay_trend"] == pytest.approx((3 + 2 + 0) / 3 - 0)
    assert out["revolving_months"] == 2


def test_trend_is_positive_when_recent_months_are_higher(clean_rows):
    clean_rows.loc[0, config.BILL_AMT] = [9_000, 9_000, 9_000, 1_000, 1_000, 1_000]
    out = engineered_features(clean_rows).loc[0]
    assert out["util_trend"] == pytest.approx(0.8)
    assert out["util_volatility"] > 0


def test_paid_full_all_flag(clean_rows):
    clean_rows.loc[0, config.PAY_AMT] = 10_000
    out = engineered_features(clean_rows)
    assert out["paid_full_all"].tolist() == [1, 0]


def test_build_features_drops_id_and_target(clean_rows):
    clean_rows["id"] = [1, 2]
    clean_rows[config.TARGET] = [0, 1]
    for col in ["age", *config.CATEGORICAL_COLS]:
        clean_rows[col] = 1
    out = build_features(clean_rows)
    assert "id" not in out and config.TARGET not in out
    assert out.shape[1] == len(config.CATEGORICAL_COLS) + len(config.RAW_NUMERIC_COLS) + 13

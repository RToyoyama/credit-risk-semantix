import numpy as np
import pandas as pd
import pytest

from credit_risk import config
from credit_risk.features import behavior_features, delay_months, payment_ratio, utilization


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

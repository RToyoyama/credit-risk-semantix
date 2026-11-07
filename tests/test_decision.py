import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from credit_risk import config
from credit_risk.cleaning import clean
from credit_risk.decision import calibrated_pipeline


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

import pandas as pd
import pytest

from credit_risk import config
from credit_risk.cleaning import clean
from credit_risk.models import CATALOG, build_pipeline


@pytest.fixture
def clean_big(raw_df) -> pd.DataFrame:
    return clean(pd.concat([raw_df] * 60, ignore_index=True))


@pytest.mark.parametrize("name", list(CATALOG))
def test_every_model_fits_and_predicts_probabilities(clean_big, name):
    pipe = build_pipeline(name).fit(clean_big, clean_big[config.TARGET])
    proba = pipe.predict_proba(clean_big)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_linear_model_gets_scaling_and_status_as_category(clean_big):
    pipe = build_pipeline("logistic").fit(clean_big, clean_big[config.TARGET])
    names = pipe.named_steps["prep"].named_steps["prep"].get_feature_names_out()
    assert any(n.startswith("pay_status_1_") for n in names)


def test_tree_model_keeps_status_numeric_and_unscaled(clean_big):
    pipe = build_pipeline("xgboost").fit(clean_big, clean_big[config.TARGET])
    prep = pipe.named_steps["prep"]
    out = prep.transform(clean_big)
    assert "pay_status_1" in out
    assert out["limit_bal"].tolist() == clean_big["limit_bal"].tolist()


def test_factory_returns_new_instances():
    assert CATALOG["xgboost"].factory() is not CATALOG["xgboost"].factory()


@pytest.mark.parametrize("name", ["voting_xgb_rf", "stacking_xgb_rf_logit"])
def test_ensembles_fit_and_return_probabilities(clean_big, name):
    from credit_risk.models import build_ensemble

    model = build_ensemble(name).fit(clean_big, clean_big[config.TARGET])
    proba = model.predict_proba(clean_big)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_soft_voting_is_mean_of_members(clean_big):
    import numpy as np

    from credit_risk.models import build_ensemble

    y = clean_big[config.TARGET]
    vote = build_ensemble("voting_xgb_rf").fit(clean_big, y)
    members = [est.predict_proba(clean_big)[:, 1] for est in vote.estimators_]
    assert np.allclose(vote.predict_proba(clean_big)[:, 1], np.mean(members, axis=0))

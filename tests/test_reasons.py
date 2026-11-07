import numpy as np
import pandas as pd
import pytest

from credit_risk import config
from credit_risk.cleaning import clean
from credit_risk.models import final_pipeline
from credit_risk.preprocessing import feature_columns
from credit_risk.reasons import ReasonExplainer, concept_of, reason_phrase


@pytest.fixture(scope="module")
def fitted(request):
    raw = request.getfixturevalue("raw_df_module")
    df = clean(pd.concat([raw] * 60, ignore_index=True))
    rng = np.random.default_rng(0)
    df["pay_status_1"] = rng.integers(-2, 4, len(df))
    df[config.TARGET] = (df["pay_status_1"] >= 2).astype(int) | (rng.random(len(df)) < 0.1)
    model = final_pipeline().fit(df, df[config.TARGET])
    return model, df.drop(columns=[config.TARGET])


@pytest.fixture(scope="module")
def raw_df_module():
    data = {
        "ID": [1, 2, 3, 4],
        "LIMIT_BAL": [20_000, 50_000, 120_000, 500_000],
        "SEX": [1, 2, 2, 1],
        "EDUCATION": [1, 2, 3, 4],
        "MARRIAGE": [1, 2, 3, 1],
        "AGE": [21, 35, 47, 79],
        **{c: [-2, -1, 0, 2] for c in config.PAY_STATUS_COLS},
        **{c: [-500, 0, 3_000, 90_000] for c in config.BILL_AMT_COLS},
        **{c: [0, 100, 2_000, 5_000] for c in config.PAY_AMT_COLS},
        config.RAW_TARGET: [0, 0, 1, 1],
    }
    return pd.DataFrame(data)[config.RAW_COLUMNS]


def test_every_model_column_has_a_concept():
    num, cat = feature_columns(exclude=tuple(config.MODEL_EXCLUDE))
    for col in num + [f"{c}_1" for c in cat]:
        assert concept_of(col)


def test_contributions_are_additive(fitted):
    model, X = fitted
    exp = ReasonExplainer(model)
    contrib = exp.contributions(X)
    p = model.predict_proba(X)[:, 1]
    logit = np.log(p / (1 - p))
    assert np.abs(contrib.sum(axis=1).to_numpy() + exp.base_value - logit).max() < 1e-4


def test_native_treeshap_matches_shap_library(fitted):
    import shap

    model, X = fitted
    ours = ReasonExplainer(model).contributions(X.head(50))
    Xt = model.named_steps["prep"].transform(X.head(50))
    lib = shap.TreeExplainer(model.named_steps["model"]).shap_values(Xt)
    assert np.allclose(ours.to_numpy(), lib, atol=1e-5)


def test_reasons_are_positive_sorted_and_at_most_three(fitted):
    model, X = fitted
    for reasons in ReasonExplainer(model).explain(X):
        values = [r["contribuicao"] for r in reasons]
        assert len(values) <= config.N_REASONS
        assert all(v >= config.MIN_REASON_CONTRIBUTION for v in values)
        assert values == sorted(values, reverse=True)


def test_reasons_are_stable_and_ignore_excluded_attributes(fitted):
    model, X = fitted
    exp = ReasonExplainer(model)
    a = exp.explain(X.head(20))
    changed = X.head(20).copy()
    changed["age"] = 99
    changed["sex"] = 2
    assert exp.explain(changed) == a


def test_reason_phrase_uses_client_values():
    row = pd.Series({"pay_status_1": 2, "util_recent": 0.95, "limit_bal": 20_000})
    assert reason_phrase("atraso recente", row) == "2 meses de atraso no último pagamento"
    assert reason_phrase("uso do limite", row) == "usa 95% do limite"
    assert reason_phrase("limite de crédito", row) == "limite de crédito de NT$ 20 mil"
    assert reason_phrase("uso do limite", pd.Series({"util_recent": 0.02})).startswith(
        "cartão quase sem uso"
    )
    assert reason_phrase("valor da fatura", pd.Series({"bill_amt_1": 0})).startswith(
        "fatura zerada"
    )
    assert reason_phrase("valor da fatura", pd.Series({"bill_amt_1": 357})) == "fatura de NT$ 357"


def test_score_clients_includes_reasons(fitted):
    from credit_risk.scoring import ScoringParams, score_clients

    model, X = fitted
    params = ScoringParams(band_cuts={"A": 600, "B": 550, "C": 500, "D": 450}, ev_cut=0.0)
    out = score_clients(model, X.head(10), params, ReasonExplainer(model))
    assert all(isinstance(m, list) and m for m in out["motivos"])


def test_tiny_contributions_are_not_reasons(fitted):
    model, X = fitted
    exp = ReasonExplainer(model)
    concepts = exp.concept_contributions(X)
    for row, reasons in zip(concepts.itertuples(index=False), exp.explain(X), strict=True):
        material = [v for v in row if v >= config.MIN_REASON_CONTRIBUTION]
        assert len(reasons) == min(len(material), config.N_REASONS)

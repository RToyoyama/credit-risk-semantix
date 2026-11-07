import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import API_VERSION, create_app
from api.schemas import EXAMPLE
from credit_risk import config
from credit_risk.cleaning import clean
from credit_risk.models import final_pipeline
from credit_risk.reasons import ReasonExplainer
from credit_risk.scoring import ScoringParams, score_clients

PARAMS = ScoringParams(band_cuts={"A": 575, "B": 532, "C": 479, "D": 375}, ev_cut=4500.0)


@pytest.fixture(scope="module")
def model():
    raw = {
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
    df = clean(pd.concat([pd.DataFrame(raw)[config.RAW_COLUMNS]] * 80, ignore_index=True))
    rng = np.random.default_rng(0)
    df["pay_status_1"] = rng.integers(-2, 4, len(df))
    df[config.TARGET] = ((df["pay_status_1"] >= 2) | (rng.random(len(df)) < 0.1)).astype(int)
    return final_pipeline().fit(df.drop(columns=[config.TARGET]), df[config.TARGET])


@pytest.fixture(scope="module")
def client(model):
    app = create_app(model=model, params=PARAMS, metadata={"versao": "teste"})
    with TestClient(app) as c:
        yield c


def _client(**overrides):
    return {**EXAMPLE, **overrides}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info_exposes_versions_cuts_and_exclusions(client):
    info = client.get("/model-info").json()
    assert info["versao_api"] == API_VERSION
    assert info["versao_modelo"] == "teste"
    assert info["cortes_faixas"] == PARAMS.band_cuts
    assert set(info["variaveis_excluidas"]) == {"sex", "marriage", "age"}
    assert info["premissas"]["lgd"] == config.LGD


def test_score_contract(client):
    body = client.post("/score", json=EXAMPLE).json()
    assert body["id"] == EXAMPLE["id"]
    assert 0 <= body["pd"] <= 1
    assert 0 <= body["score"] <= 1000
    assert body["faixa"] in config.BAND_SHARES
    assert body["acao"] == config.BAND_ACTIONS[body["faixa"]]
    assert body["perda_esperada"] == pytest.approx(body["pd"] * config.LGD * body["ead"])
    assert isinstance(body["motivos"], list) and body["motivos"]
    assert body["versao_api"] == API_VERSION


def test_id_is_optional(client):
    payload = _client()
    payload.pop("id")
    assert client.post("/score", json=payload).json()["id"] is None


@pytest.mark.parametrize(
    "payload",
    [
        _client(sex=1),
        _client(age=30),
        _client(marriage=2),
        _client(pay_status_1=15),
        _client(pay_status_2=-3),
        _client(education=5),
        _client(limit_bal=0),
        _client(pay_amt_1=-10),
        {k: v for k, v in EXAMPLE.items() if k != "bill_amt_3"},
        _client(limit_bal="muito"),
    ],
)
def test_invalid_input_returns_422(client, payload):
    r = client.post("/score", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]


def test_negative_bill_is_accepted_and_ead_is_zero(client):
    body = client.post("/score", json=_client(bill_amt_1=-2_000)).json()
    assert body["ead"] == 0 and body["perda_esperada"] == 0


def test_batch_preserves_order_and_size(client):
    clients = [_client(id=i, pay_status_1=s) for i, s in enumerate([-1, 0, 2, 3, -2])]
    body = client.post("/score/batch", json={"clientes": clients}).json()
    assert body["n"] == 5
    assert [r["id"] for r in body["resultados"]] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("size", [0, 1001])
def test_batch_size_limits(client, size):
    assert client.post("/score/batch", json={"clientes": [EXAMPLE] * size}).status_code == 422


def test_api_matches_scoring_module_exactly(client, model):
    clients = [
        _client(id=i, pay_status_1=s, bill_amt_1=b)
        for i, (s, b) in enumerate([(-1, 500), (0, 20_000), (2, 48_500), (3, 90_000)])
    ]
    api = pd.DataFrame(client.post("/score/batch", json={"clientes": clients}).json()["resultados"])
    direct = score_clients(model, pd.DataFrame(clients), PARAMS, ReasonExplainer(model))
    assert np.allclose(api["pd"], direct["pd"])
    assert api["score"].tolist() == direct["score"].tolist()
    assert api["faixa"].tolist() == direct["faixa"].tolist()
    assert api["prioritario"].tolist() == direct["prioritario"].tolist()
    assert api["motivos"].tolist() == direct["motivos"].tolist()

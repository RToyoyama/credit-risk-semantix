"""Monitoramento de estabilidade da população (PSI) do score e das variáveis.

A referência (faixas e proporções) é definida uma vez, com o treino, e salva.
``run_monitoring`` recebe um lote de clientes e devolve o relatório: é a
tarefa que um agendador (cron, GitHub Actions, Airflow) chamaria todo mês.

Uso: ``python -m credit_risk.monitoring`` (ou ``make monitor``).
"""

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

from credit_risk import config
from credit_risk.features import engineered_features
from credit_risk.scoring import assign_band, load_params, score_clients

logger = logging.getLogger(__name__)


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = config.PSI_EPSILON) -> float:
    """Population Stability Index entre duas distribuições por faixa.

    PSI = soma de (atual − esperado) × ln(atual ÷ esperado). Exemplo: 50%/50%
    → 70%/30% dá 0,067 + 0,102 = 0,17.

    Args:
        expected: Proporções da referência (somam 1).
        actual: Proporções do lote (somam 1).
        eps: Piso para faixas vazias (evita log de zero).

    Returns:
        PSI (0 = idênticas).
    """
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    """Leitura de mercado do PSI.

    Args:
        value: PSI.

    Returns:
        "estável", "atenção" ou "investigar".
    """
    if value > config.PSI_THRESHOLDS["investigar"]:
        return "investigar"
    if value > config.PSI_THRESHOLDS["atenção"]:
        return "atenção"
    return "estável"


def _proportions(values: pd.Series, ref: dict[str, Any]) -> np.ndarray:
    """Proporção de ``values`` em cada faixa (ou categoria) da referência."""
    if "categories" in ref:
        cats = ref["categories"]
        clipped = values.clip(upper=cats[-1]).clip(lower=cats[0])
        counts = clipped.value_counts().reindex(cats, fill_value=0)
    else:
        edges = np.array(ref["edges"], dtype=float)
        idx = np.clip(
            np.searchsorted(edges, values.to_numpy(dtype=float), side="right"), 0, len(edges)
        )
        counts = pd.Series(np.bincount(idx, minlength=len(edges) + 1))
    return (counts / counts.sum()).to_numpy()


def fit_reference(data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Define as faixas e proporções de referência (com o treino).

    Variáveis numéricas: cortes nos decis. Categóricas: os valores observados
    (os extremos acumulam o que estiver além deles).

    Args:
        data: Uma coluna por variável monitorada.

    Returns:
        Referência por variável.
    """
    ref: dict[str, dict[str, Any]] = {}
    for col in config.MONITORED_NUMERIC:
        qs = np.unique(np.quantile(data[col], np.linspace(0, 1, config.PSI_BINS + 1)[1:-1]))
        ref[col] = {"edges": qs.tolist()}
    for col in config.MONITORED_CATEGORICAL:
        ref[col] = {"categories": sorted(int(v) for v in data[col].unique())}
    for col, spec in ref.items():
        spec["props"] = _proportions(data[col], spec).tolist()
    return ref


def monitoring_frame(model: object, clients: pd.DataFrame) -> pd.DataFrame:
    """Monta as colunas monitoradas de um lote (score, PD e variáveis de entrada).

    Args:
        model: Pipeline final ajustado.
        clients: Dados limpos do lote.

    Returns:
        Uma coluna por variável monitorada.
    """
    scored = score_clients(model, clients, load_params())
    eng = engineered_features(clients)
    return pd.DataFrame(
        {
            "score": scored["score"].to_numpy(),
            "pd": scored["pd"].to_numpy(),
            "util_recent": eng["util_recent"].to_numpy(),
            "pay_ratio_recent": eng["pay_ratio_recent"].to_numpy(),
            "delay_months": eng["delay_months"].to_numpy(),
            "pay_status_1": clients["pay_status_1"].to_numpy(),
            "bill_amt_1": clients["bill_amt_1"].to_numpy(),
            "limit_bal": clients["limit_bal"].to_numpy(),
            "age": clients["age"].to_numpy(),
        }
    )


def run_monitoring(
    clients: pd.DataFrame, model: object, reference: dict[str, dict[str, Any]], name: str = "lote"
) -> dict[str, Any]:
    """Relatório de estabilidade de um lote contra a referência.

    Além do PSI por variável, confere dois indicadores complementares (o PSI
    é pouco sensível a mudanças concentradas na cauda de risco): a PD média do
    lote e a fração de clientes na faixa E, ambas contra a referência do treino.

    Args:
        clients: Dados limpos do lote.
        model: Pipeline final ajustado.
        reference: Saída de ``fit_reference``.
        name: Nome do lote.

    Returns:
        PSI e status por variável, pior status e as variáveis em alerta.
    """
    frame = monitoring_frame(model, clients)
    params = load_params()
    band_e = float((assign_band(frame["score"], params.band_cuts) == "E").mean())
    ref_ind = reference["_indicadores"]
    indicators = {
        "pd_media": {
            "valor": round(float(frame["pd"].mean()), 4),
            "referencia": ref_ind["pd_media"],
            "razao": round(float(frame["pd"].mean()) / ref_ind["pd_media"], 3),
        },
        "fracao_faixa_e": {
            "valor": round(band_e, 4),
            "referencia": ref_ind["fracao_faixa_e"],
            "razao": round(band_e / ref_ind["fracao_faixa_e"], 3),
        },
    }
    indicators["pd_media"]["status"] = (
        "investigar" if indicators["pd_media"]["razao"] > config.PD_MEAN_RATIO_MAX else "estável"
    )
    indicators["fracao_faixa_e"]["status"] = (
        "investigar"
        if indicators["fracao_faixa_e"]["razao"] > config.BAND_E_RATIO_MAX
        else "estável"
    )
    variables = {}
    for col, spec in reference.items():
        if col.startswith("_"):
            continue
        value = psi(spec["props"], _proportions(frame[col], spec))
        variables[col] = {"psi": round(value, 4), "status": status(value)}
    order = ["estável", "atenção", "investigar"]
    everything = {**variables, **indicators}
    worst = max((v["status"] for v in everything.values()), key=order.index)
    return {
        "lote": name,
        "clientes": len(clients),
        "status_geral": worst,
        "alertas": [c for c, v in everything.items() if v["status"] != "estável"],
        "indicadores": indicators,
        "variaveis": variables,
    }


def drift_scenarios(holdout: pd.DataFrame, seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    """Lotes simulados com a população deslocada (subconjuntos do holdout).

    Args:
        holdout: Dados limpos do holdout.
        seed: Semente do reamostrador do cenário de crise.

    Returns:
        Nome do cenário → lote.
    """
    late = holdout["pay_status_1"] >= 1
    n_extra = int(late.sum())
    rng = np.random.default_rng(seed)
    extra = holdout[late].iloc[rng.integers(0, late.sum(), n_extra)]
    return {
        "holdout (mesma população)": holdout,
        "só clientes até 25 anos": holdout[holdout["age"] <= 25],
        "só limite até NT$ 50 mil": holdout[holdout["limit_bal"] <= 50_000],
        "crise: dobra a fração de atrasados": pd.concat([holdout, extra], ignore_index=True),
    }


def build_reference(model: object) -> dict[str, dict[str, Any]]:
    """Referência a partir do treino (score e PD fora do fold) e salva em disco.

    Args:
        model: Pipeline final ajustado (usado só para as variáveis de entrada).

    Returns:
        A referência.
    """
    from credit_risk.scoring import pd_to_score
    from credit_risk.split import load_train

    train = load_train()
    oof = pd.read_parquet(config.OOF_PREDICTIONS).set_index("id").loc[train["id"]]
    frame = monitoring_frame(model, train.drop(columns=[config.TARGET]))
    frame["pd"] = oof["pd"].to_numpy()
    frame["score"] = pd_to_score(frame["pd"])
    reference = fit_reference(frame)
    reference["_indicadores"] = {
        "pd_media": round(float(frame["pd"].mean()), 4),
        "fracao_faixa_e": round(
            float((assign_band(frame["score"], load_params().band_cuts) == "E").mean()), 4
        ),
    }
    config.MONITORING_REFERENCE.write_text(json.dumps(reference, indent=2) + "\n")
    return reference


def main() -> list[dict[str, Any]]:
    """Gera a referência e o relatório do holdout e dos cenários de drift.

    Returns:
        Lista de relatórios (também salva em ``config.MONITORING_REPORT``).
    """
    model = joblib.load(config.FINAL_MODEL_PATH)
    reference = build_reference(model)
    holdout = pd.read_parquet(config.HOLDOUT_PARQUET).drop(columns=[config.TARGET])
    reports = [run_monitoring(b, model, reference, n) for n, b in drift_scenarios(holdout).items()]
    config.MONITORING_REPORT.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
    return reports


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for r in main():
        print(
            f"{r['lote']:<38} {r['clientes']:>5} clientes | "
            f"{r['status_geral']:<10} | alertas: {r['alertas']}"
        )

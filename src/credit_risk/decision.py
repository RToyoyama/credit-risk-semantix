"""Calibração das probabilidades e política de decisão por custo.

Parte A (calibração): o XGBoost ajustado já sai quase calibrado (Etapa 10);
aqui se testa se uma correção sigmoid ou isotônica melhora o Brier.

Parte B (decisão): para cada cliente, compara o que se evita ao agir com o
que custa agir. As premissas de negócio ficam em ``config.py``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from credit_risk import config
from credit_risk.models import build_pipeline

CALIBRATION_METHODS = ["none", "sigmoid", "isotonic"]


def calibrated_pipeline(method: str = "none") -> Pipeline | CalibratedClassifierCV:
    """Modelo final (XGBoost ajustado), com ou sem correção das probabilidades.

    A correção é aprendida em partes do treino que o modelo não viu
    (validação cruzada interna de 5 folds), sem vazamento.

    Args:
        method: ``"none"``, ``"sigmoid"`` (Platt) ou ``"isotonic"``.

    Returns:
        Estimador não ajustado.

    Raises:
        ValueError: Se o método for desconhecido.
    """
    base = build_pipeline(config.FINAL_MODEL, tuned=True)
    if method == "none":
        return base
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Método de calibração desconhecido: {method}")
    cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.SEED)
    return CalibratedClassifierCV(base, method=method, cv=cv)


# ── Parte B: política de decisão por custo ──────────────────


@dataclass(frozen=True)
class Premises:
    """Premissas de negócio do cálculo de custo (valores em NT$).

    Attributes:
        lgd: Fração da dívida perdida quando o cliente dá calote.
        efficacy: Fração dessa perda que a ação preventiva evita.
        action_cost: Custo de acionar um cliente.
        friction_cost: Margem perdida ao acionar quem pagaria.
    """

    lgd: float = config.LGD
    efficacy: float = config.ACTION_EFFICACY
    action_cost: float = config.ACTION_COST
    friction_cost: float = config.FRICTION_COST


def exposure(df: pd.DataFrame) -> pd.Series:
    """Exposição no calote (EAD): fatura mais recente, com saldo negativo como zero.

    Args:
        df: Dados limpos.

    Returns:
        EAD por cliente, em NT$.
    """
    return df["bill_amt_1"].clip(lower=0).astype(float).rename("ead")


def realized_value(y: np.ndarray, ead: np.ndarray, prem: Premises) -> np.ndarray:
    """Valor de acionar cada cliente, dado o que de fato aconteceu.

    Caloteiro acionado: LGD × EAD × eficácia − custo. Bom pagador acionado:
    −custo − atrito.

    Args:
        y: Alvo real (1 = calote).
        ead: Exposição de cada cliente.
        prem: Premissas.

    Returns:
        Valor (NT$) de acionar cada cliente, em relação a não acionar.
    """
    y = np.asarray(y)
    ead = np.asarray(ead, dtype=float)
    saved = prem.lgd * ead * prem.efficacy - prem.action_cost
    return np.where(y == 1, saved, -prem.action_cost - prem.friction_cost)


def expected_value(pd_: np.ndarray, ead: np.ndarray, prem: Premises) -> np.ndarray:
    """Valor esperado de acionar cada cliente, usando a probabilidade de calote.

    Args:
        pd_: Probabilidade de default prevista.
        ead: Exposição de cada cliente.
        prem: Premissas.

    Returns:
        Valor esperado (NT$) de acionar cada cliente.
    """
    p = np.asarray(pd_, dtype=float)
    ead = np.asarray(ead, dtype=float)
    return p * prem.lgd * ead * prem.efficacy - prem.action_cost - (1 - p) * prem.friction_cost


def policy_result(act: np.ndarray, y: np.ndarray, ead: np.ndarray, prem: Premises) -> dict:
    """Resultado de uma política (quem é acionado) contra o status quo.

    Args:
        act: Máscara booleana de clientes acionados.
        y: Alvo real.
        ead: Exposição.
        prem: Premissas.

    Returns:
        Acionados, calotes alcançados, valor total e perda total sem ação.
    """
    act = np.asarray(act, dtype=bool)
    y = np.asarray(y)
    value = realized_value(y, ead, prem)
    loss = prem.lgd * np.asarray(ead, dtype=float) * (y == 1)
    return {
        "acionados": int(act.sum()),
        "pct_acionados": float(act.mean()),
        "calotes_alcancados": int((act & (y == 1)).sum()),
        "pct_calotes_alcancados": float((act & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "valor": float(value[act].sum()),
        "perda_sem_acao": float(loss.sum()),
    }


def threshold_curve(
    pd_: np.ndarray, y: np.ndarray, ead: np.ndarray, prem: Premises, step: float = 0.01
) -> pd.DataFrame:
    """Valor da política "acionar se PD >= limiar" para cada limiar.

    Args:
        pd_: Probabilidade prevista.
        y: Alvo real.
        ead: Exposição.
        prem: Premissas.
        step: Passo da grade de limiares.

    Returns:
        Uma linha por limiar.
    """
    pd_ = np.asarray(pd_)
    rows = [
        {"limiar": t, **policy_result(pd_ >= t, y, ead, prem)}
        for t in np.round(np.arange(step, 1, step), 4)
    ]
    return pd.DataFrame(rows)


def delay_rules(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Regras fixas simples, sem modelo: os adversários definidos na Etapa 1.

    Args:
        df: Dados limpos.

    Returns:
        Nome da regra → máscara de clientes acionados.
    """
    recent = df["pay_status_1"].clip(lower=0).to_numpy()
    months = (df[config.PAY_STATUS] > 0).sum(axis=1).to_numpy()
    rules = {f"atraso em set >= {k} meses": recent >= k for k in (1, 2, 3)}
    rules.update({f"meses com atraso >= {k}": months >= k for k in (1, 2, 3, 4)})
    return rules


def bootstrap_diff(
    value_a: np.ndarray,
    value_b: np.ndarray,
    n_boot: int = config.N_BOOT,
    level: float = config.CI_LEVEL,
    seed: int = config.SEED,
) -> tuple[float, float, float]:
    """IC bootstrap da diferença de valor total entre duas políticas.

    Reamostra clientes (com reposição) e recalcula a diferença, mantendo o
    pareamento: o mesmo cliente entra nas duas políticas.

    Args:
        value_a: Valor por cliente da política A (0 se não acionado).
        value_b: Valor por cliente da política B.
        n_boot: Reamostragens.
        level: Nível de confiança.
        seed: Semente.

    Returns:
        Tupla ``(diferença, limite_inferior, limite_superior)`` de A − B.
    """
    diff = np.asarray(value_a) - np.asarray(value_b)
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].sum() for _ in range(n_boot)])
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(boots, [tail, 100 - tail])
    return float(diff.sum()), float(low), float(high)


def capacity_mask(
    score: np.ndarray, capacity: float, eligible: np.ndarray | None = None
) -> np.ndarray:
    """Aciona os ``capacity`` × n clientes de maior score (entre os elegíveis).

    Args:
        score: Prioridade de cada cliente (maior = acionar primeiro).
        capacity: Fração máxima da carteira acionada.
        eligible: Máscara de quem pode ser acionado (None = todos).

    Returns:
        Máscara booleana dos acionados.
    """
    score = np.asarray(score, dtype=float)
    n_max = int(np.floor(capacity * len(score)))
    allowed = np.ones(len(score), dtype=bool) if eligible is None else np.asarray(eligible, bool)
    ranked = np.where(allowed, score, -np.inf)
    order = np.argsort(-ranked, kind="stable")[:n_max]
    mask = np.zeros(len(score), dtype=bool)
    mask[order[np.isfinite(ranked[order])]] = True
    return mask


def rule_priority(df: pd.DataFrame) -> np.ndarray:
    """Prioridade de um analista sem modelo: mais atrasado primeiro, depois maior fatura.

    Ordena por atraso em setembro, depois por meses com atraso no semestre e,
    no empate, pela maior fatura (EAD).

    Args:
        df: Dados limpos.

    Returns:
        Score de prioridade (maior = acionar primeiro).
    """
    recent = df["pay_status_1"].clip(lower=0).to_numpy(dtype=float)
    months = (df[config.PAY_STATUS] > 0).sum(axis=1).to_numpy(dtype=float)
    ead = exposure(df).to_numpy()
    return recent * 1e12 + months * 1e10 + ead

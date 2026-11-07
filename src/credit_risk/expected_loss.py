"""Perda esperada (EL = PD × LGD × EAD) por cliente e por carteira.

A EAD (exposição no calote) vem de ``decision.exposure``: fatura mais recente,
com saldo negativo como zero. A LGD é premissa de negócio (``config.LGD``).
"""

import numpy as np
import pandas as pd

from credit_risk import config


def expected_loss(pd_: np.ndarray, ead: np.ndarray, lgd: float = config.LGD) -> np.ndarray:
    """Perda esperada de cada cliente.

    Exemplo: PD 20%, EAD NT$ 10.000, LGD 60% → 0,20 × 0,60 × 10.000 = NT$ 1.200.

    Args:
        pd_: Probabilidade de default.
        ead: Exposição no calote (NT$).
        lgd: Fração da exposição perdida no calote.

    Returns:
        EL por cliente (NT$).
    """
    return np.asarray(pd_, dtype=float) * lgd * np.asarray(ead, dtype=float)


def top_mask(score: np.ndarray, share: float) -> np.ndarray:
    """Marca os ``share`` × n clientes de maior score.

    Args:
        score: Critério de ordenação (maior = mais prioritário).
        share: Fração da carteira marcada.

    Returns:
        Máscara booleana.
    """
    score = np.asarray(score, dtype=float)
    n = int(np.floor(share * len(score)))
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-score, kind="stable")[:n]] = True
    return mask


def ranking_comparison(
    pd_: np.ndarray,
    ead: np.ndarray,
    y: np.ndarray,
    share: float = config.CAPACITY,
    lgd: float = config.LGD,
) -> dict[str, float]:
    """Prioridade por EL × por PD: quem muda e quanto da perda cada uma concentra.

    Args:
        pd_: Probabilidade de default.
        ead: Exposição.
        y: Alvo real (para a perda realizada).
        share: Tamanho do topo comparado (ex.: 10%).
        lgd: LGD.

    Returns:
        Tamanho do topo, clientes em comum e as frações de perda esperada e
        realizada capturadas por cada ranking.
    """
    el = expected_loss(pd_, ead, lgd)
    realized = lgd * np.asarray(ead, dtype=float) * (np.asarray(y) == 1)
    by_el, by_pd = top_mask(el, share), top_mask(pd_, share)
    return {
        "topo": int(by_el.sum()),
        "em_comum": int((by_el & by_pd).sum()),
        "mudam_de_prioridade": int((by_el & ~by_pd).sum()),
        "el_capturada_por_el": float(el[by_el].sum() / el.sum()),
        "el_capturada_por_pd": float(el[by_pd].sum() / el.sum()),
        "perda_real_capturada_por_el": float(realized[by_el].sum() / realized.sum()),
        "perda_real_capturada_por_pd": float(realized[by_pd].sum() / realized.sum()),
    }


def concentration_curve(
    values: np.ndarray, order_by: np.ndarray, points: int = 100
) -> pd.DataFrame:
    """Fração acumulada de ``values`` capturada pelos x% primeiros segundo ``order_by``.

    Args:
        values: O que se acumula (ex.: perda realizada).
        order_by: Critério de ordenação (maior primeiro).
        points: Número de pontos da curva.

    Returns:
        Colunas ``fracao_clientes`` e ``fracao_capturada``.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(-np.asarray(order_by, dtype=float), kind="stable")
    cum = np.cumsum(values[order]) / values.sum()
    idx = np.linspace(0, len(values) - 1, points).astype(int)
    return pd.DataFrame({"fracao_clientes": (idx + 1) / len(values), "fracao_capturada": cum[idx]})

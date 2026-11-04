"""Validação cruzada repetida e comparação pareada entre pipelines.

Todo experimento compara duas variantes **nos mesmos folds**, para que a
diferença reflita a mudança testada e não a sorte da divisão dos dados.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from credit_risk import config

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier": "neg_brier_score",
}


def make_cv(seed: int = config.SEED) -> RepeatedStratifiedKFold:
    """Validação cruzada padrão do projeto: 5 folds × 3 repetições.

    Args:
        seed: Semente da divisão.

    Returns:
        Objeto de validação cruzada.
    """
    return RepeatedStratifiedKFold(
        n_splits=config.CV_SPLITS, n_repeats=config.CV_REPEATS, random_state=seed
    )


def cv_scores(
    estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series, seed: int = config.SEED
) -> pd.DataFrame:
    """Métricas de cada fold (Brier com sinal positivo: quanto menor, melhor).

    Args:
        estimator: Pipeline completo (pré-processamento + modelo).
        X: Features.
        y: Alvo.
        seed: Semente da divisão (a mesma semente gera os mesmos folds).

    Returns:
        Uma linha por fold, uma coluna por métrica.
    """
    res = cross_validate(estimator, X, y, cv=make_cv(seed), scoring=SCORING, n_jobs=-1)
    out = pd.DataFrame({name: res[f"test_{name}"] for name in SCORING})
    out["brier"] = -out["brier"]
    return out


def paired_comparison(
    base: pd.DataFrame, candidate: pd.DataFrame, metric: str = "pr_auc"
) -> dict[str, float]:
    """Compara duas variantes fold a fold.

    Args:
        base: Métricas por fold da variante atual.
        candidate: Métricas por fold da variante testada (mesmos folds).
        metric: Métrica comparada (maior = melhor, exceto ``brier``).

    Returns:
        Médias, diferença média, desvio da diferença e win rate do candidato.
    """
    diff = candidate[metric].to_numpy() - base[metric].to_numpy()
    if metric == "brier":
        diff = -diff
    return {
        "base": float(base[metric].mean()),
        "candidato": float(candidate[metric].mean()),
        "ganho_medio": float(diff.mean()),
        "desvio_ganho": float(diff.std(ddof=1)),
        "win_rate": float(np.mean(diff > 0)),
    }

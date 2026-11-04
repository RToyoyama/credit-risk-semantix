"""Validação cruzada repetida e comparação pareada entre pipelines.

Todo experimento compara duas variantes **nos mesmos folds**, para que a
diferença reflita a mudança testada e não a sorte da divisão dos dados.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, roc_curve
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from credit_risk import config


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Estatística KS: maior distância entre as curvas acumuladas de bons e maus.

    Equivale ao máximo de (taxa de verdadeiros positivos − taxa de falsos
    positivos) ao longo de todos os pontos de corte.

    Args:
        y_true: Alvo binário.
        y_score: Probabilidade (ou score) prevista para a classe positiva.

    Returns:
        KS entre 0 (não separa) e 1 (separa perfeitamente).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "ks": make_scorer(ks_statistic, response_method="predict_proba"),
    "brier": "neg_brier_score",
    "accuracy": "accuracy",
}


LOWER_IS_BETTER = {"brier"}


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
    if metric in LOWER_IS_BETTER:
        diff = -diff
    return {
        "base": float(base[metric].mean()),
        "candidato": float(candidate[metric].mean()),
        "ganho_medio": float(diff.mean()),
        "desvio_ganho": float(diff.std(ddof=1)),
        "win_rate": float(np.mean(diff > 0)),
    }


def summarize(
    fold_scores: dict[str, pd.DataFrame], reference: str, metric: str = "pr_auc"
) -> pd.DataFrame:
    """Média e desvio de cada métrica por modelo e comparação pareada com uma referência.

    Args:
        fold_scores: Métricas por fold de cada modelo (mesmos folds).
        reference: Modelo contra o qual os demais são comparados.
        metric: Métrica da comparação pareada.

    Returns:
        Uma linha por modelo, ordenada pela média de ``metric``.
    """
    rows = []
    for name, scores in fold_scores.items():
        row = {"modelo": name}
        for col in scores.columns:
            row[f"{col}_media"] = scores[col].mean()
            row[f"{col}_desvio"] = scores[col].std(ddof=1)
        if name != reference:
            comp = paired_comparison(fold_scores[reference], scores, metric)
            row[f"ganho_vs_{reference}"] = comp["ganho_medio"]
            row[f"win_rate_vs_{reference}"] = comp["win_rate"]
        rows.append(row)
    ascending = metric in LOWER_IS_BETTER
    return (
        pd.DataFrame(rows)
        .sort_values(f"{metric}_media", ascending=ascending)
        .reset_index(drop=True)
    )


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Probabilidade média prevista × taxa real, em faixas de mesmo tamanho.

    Num modelo calibrado, a faixa com média prevista de 30% tem cerca de 30%
    de defaults.

    Args:
        y_true: Alvo binário.
        y_prob: Probabilidade prevista.
        n_bins: Número de faixas (quantis da probabilidade prevista).

    Returns:
        Uma linha por faixa: ``prevista``, ``observada`` e ``n``.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df["faixa"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    return (
        df.groupby("faixa")
        .agg(prevista=("p", "mean"), observada=("y", "mean"), n=("y", "size"))
        .reset_index()
    )


def oof_predictions(
    estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series, seed: int = config.SEED
) -> np.ndarray:
    """Probabilidade out-of-fold: cada cliente previsto por um modelo que não o viu.

    Args:
        estimator: Pipeline completo.
        X: Features.
        y: Alvo.
        seed: Semente da divisão (5 folds estratificados).

    Returns:
        Probabilidade da classe positiva para cada cliente.
    """
    cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=seed)
    return cross_val_predict(estimator, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

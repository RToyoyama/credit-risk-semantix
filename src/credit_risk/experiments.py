"""Experimentos pareados do projeto, cada um salvo em ``reports/``.

Cada experimento compara variantes nos mesmos folds (CV 5 x 3) e aplica
o critério fixado antes de rodar: adotar só se o PR AUC subir com win
rate > ``config.WIN_RATE_MIN``.

Uso: ``python -m credit_risk.experiments`` (ou ``make experiments``).
"""

import logging

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from credit_risk import config
from credit_risk.evaluate import cv_scores, paired_comparison, summarize
from credit_risk.models import CATALOG, build_pipeline
from credit_risk.preprocessing import make_feature_pipeline
from credit_risk.split import load_train

logger = logging.getLogger(__name__)

METRICS = ["pr_auc", "roc_auc", "brier"]


def reference_models() -> dict[str, BaseEstimator]:
    """Modelos de referência dos experimentos de features (do catálogo, sem tuning).

    Returns:
        Regressão logística e XGBoost.
    """
    return {"logística": CATALOG["logistic"].factory(), "xgboost": CATALOG["xgboost"].factory()}


def model_pipeline(
    model: BaseEstimator,
    scaled: bool = True,
    engineered: bool = True,
    status_as_category: bool = False,
) -> Pipeline:
    """Pipeline completo: features + pré-processamento oficial + modelo.

    Args:
        model: Classificador.
        scaled: Padroniza os números.
        engineered: Inclui as features de resumo.
        status_as_category: Trata ``pay_status_*`` como categoria.

    Returns:
        Pipeline não ajustado.
    """
    return Pipeline(
        [
            ("prep", make_feature_pipeline(scaled, engineered, status_as_category)),
            ("model", model),
        ]
    )


def _compare(variants: dict[str, tuple[dict, dict]], name: str) -> pd.DataFrame:
    """Roda pares (base, candidato) de configurações e salva a comparação."""
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    models = reference_models()
    rows = []
    for model_name, (base_kw, cand_kw) in variants.items():
        model = models[model_name]
        base = cv_scores(model_pipeline(model, **base_kw), X, y)
        cand = cv_scores(model_pipeline(model, **cand_kw), X, y)
        for metric in METRICS:
            rows.append(
                {"modelo": model_name, "métrica": metric, **paired_comparison(base, cand, metric)}
            )
        logger.info("%s: %s concluído", name, model_name)
    out = decide(pd.DataFrame(rows))
    out.to_csv(config.REPORTS_DIR / f"{name}.csv", index=False)
    return out


def decide(results: pd.DataFrame) -> pd.DataFrame:
    """Marca a coluna ``adotar`` segundo o critério fixado (PR AUC + win rate).

    Args:
        results: Linhas de ``paired_comparison`` com colunas ``modelo`` e ``métrica``.

    Returns:
        A mesma tabela com a coluna ``adotar``.
    """
    is_pr = results["métrica"] == "pr_auc"
    results["adotar"] = (
        is_pr & (results["ganho_medio"] > 0) & (results["win_rate"] > config.WIN_RATE_MIN)
    )
    return results


def feature_experiment() -> pd.DataFrame:
    """Etapa 7: colunas originais × originais + features de resumo.

    Returns:
        Comparação pareada por modelo e métrica (salva em ``reports/``).
    """
    return _compare(
        {
            "logística": ({"engineered": False}, {"engineered": True}),
            "xgboost": ({"scaled": False, "engineered": False}, {"scaled": False}),
        },
        "feature_experiment",
    )


def status_category_experiment() -> pd.DataFrame:
    """Etapa 8: ``pay_status`` como número × como categoria, na logística.

    Árvores ficam de fora: elas cortam em qualquer ponto, então tratar o
    status como número não as engana.

    Returns:
        Comparação pareada (salva em ``reports/``).
    """
    return _compare(
        {"logística": ({}, {"status_as_category": True})},
        "status_category_experiment",
    )


def model_duel() -> pd.DataFrame:
    """Etapa 9: os quatro modelos (e a referência) nos mesmos 15 folds.

    Salva as métricas de cada fold (``model_duel_folds.csv``) e o resumo com a
    comparação pareada contra o líder em PR AUC (``model_duel_summary.csv``).

    Returns:
        Resumo ordenado por PR AUC.
    """
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    folds = {}
    for name in CATALOG:
        folds[name] = cv_scores(build_pipeline(name), X, y)
        logger.info("Duelo: %s concluído", name)
    long = pd.concat(
        [s.assign(modelo=n, fold=range(len(s))) for n, s in folds.items()], ignore_index=True
    )
    long.to_csv(config.REPORTS_DIR / "model_duel_folds.csv", index=False)
    leader = max((n for n in folds if n != "dummy"), key=lambda n: folds[n]["pr_auc"].mean())
    summary = summarize(folds, reference=leader)
    summary.insert(1, "nome", summary["modelo"].map(lambda n: CATALOG[n].label))
    summary.to_csv(config.REPORTS_DIR / "model_duel_summary.csv", index=False)
    return summary


EXPERIMENTS = {
    "feature_experiment": feature_experiment,
    "status_category_experiment": status_category_experiment,
    "model_duel": model_duel,
}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    chosen = sys.argv[1:] or list(EXPERIMENTS)
    for exp_name, run in ((n, EXPERIMENTS[n]) for n in chosen):
        print(f"\n== {exp_name} ==")
        print(run().round(4).to_string(index=False))

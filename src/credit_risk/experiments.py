"""Experimentos pareados do projeto, cada um salvo em ``reports/``.

Cada experimento compara variantes nos mesmos folds (CV 5 x 3) e aplica
o critério fixado antes de rodar: adotar só se o PR AUC subir com win
rate > ``config.WIN_RATE_MIN``.

Uso: ``python -m credit_risk.experiments`` (ou ``make experiments``).
"""

import logging
from collections.abc import Callable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from credit_risk import config
from credit_risk.evaluate import cv_scores, paired_comparison
from credit_risk.features import build_features
from credit_risk.split import load_train

logger = logging.getLogger(__name__)

METRICS = ["pr_auc", "roc_auc", "brier"]


def reference_models() -> dict[str, BaseEstimator]:
    """Modelos de referência para os experimentos (sem tuning).

    Returns:
        Regressão logística e XGBoost com hiperparâmetros fixos e razoáveis.
    """
    return {
        "logística": LogisticRegression(max_iter=2_000),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.SEED,
            n_jobs=1,
        ),
    }


def _simple_preprocessor(numeric: list[str]) -> ColumnTransformer:
    """Escala nos numéricos e one-hot nas categorias (versão mínima da Etapa 8)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_COLS),
        ]
    )


def _select(columns: list[str]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def pick(df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df)[columns]

    return pick


def feature_set_pipeline(model: BaseEstimator, with_engineered: bool) -> Pipeline:
    """Pipeline com as colunas originais, com ou sem as features de resumo.

    Args:
        model: Classificador.
        with_engineered: Inclui ``config.ENGINEERED_FEATURES``.

    Returns:
        Pipeline completo.
    """
    numeric = config.RAW_NUMERIC_COLS + (config.ENGINEERED_FEATURES if with_engineered else [])
    columns = config.CATEGORICAL_COLS + numeric
    return Pipeline(
        [
            ("features", FunctionTransformer(_select(columns))),
            ("prep", _simple_preprocessor(numeric)),
            ("model", model),
        ]
    )


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
    """Colunas originais × originais + features de resumo, nos mesmos folds.

    Returns:
        Tabela de comparação pareada por modelo e métrica (salva em ``reports/``).
    """
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    rows = []
    for name, model in reference_models().items():
        base = cv_scores(feature_set_pipeline(model, with_engineered=False), X, y)
        cand = cv_scores(feature_set_pipeline(model, with_engineered=True), X, y)
        for metric in METRICS:
            rows.append(
                {"modelo": name, "métrica": metric, **paired_comparison(base, cand, metric)}
            )
        logger.info("Experimento de features: %s concluído", name)
    out = decide(pd.DataFrame(rows))
    out.to_csv(config.REPORTS_DIR / "feature_experiment.csv", index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print(feature_experiment().round(4).to_string(index=False))

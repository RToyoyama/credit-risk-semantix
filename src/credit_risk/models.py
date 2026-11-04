"""Catálogo de modelos e montagem do pipeline completo de cada um.

Cada modelo recebe o pré-processamento adequado: modelos lineares com escala
e ``pay_status`` como categoria; árvores com os números como estão.
Hiperparâmetros aqui são padrões razoáveis, sem tuning (Etapa 11).
"""

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from credit_risk import config
from credit_risk.preprocessing import make_feature_pipeline


@dataclass(frozen=True)
class ModelSpec:
    """Descrição de um modelo do catálogo.

    Attributes:
        label: Nome legível.
        linear: Se True, recebe escala e status como categoria.
        factory: Função que cria o estimador (novo a cada chamada).
    """

    label: str
    linear: bool
    factory: Callable[[], BaseEstimator]


CATALOG: dict[str, ModelSpec] = {
    "dummy": ModelSpec("Referência (taxa média)", False, lambda: DummyClassifier(strategy="prior")),
    "logistic": ModelSpec("Regressão logística", True, lambda: LogisticRegression(max_iter=2_000)),
    "decision_tree": ModelSpec(
        "Árvore de Decisão",
        False,
        lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=config.SEED),
    ),
    "random_forest": ModelSpec(
        "Random Forest",
        False,
        lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, random_state=config.SEED, n_jobs=1
        ),
    ),
    "xgboost": ModelSpec(
        "XGBoost",
        False,
        lambda: XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.SEED,
            n_jobs=1,
        ),
    ),
}


def build_pipeline(name: str, model: BaseEstimator | None = None) -> Pipeline:
    """Pipeline completo (features → pré-processamento → modelo) de um modelo do catálogo.

    Args:
        name: Chave em ``CATALOG``.
        model: Estimador a usar no lugar do padrão (ex.: com hiperparâmetros ajustados).

    Returns:
        Pipeline não ajustado.
    """
    spec = CATALOG[name]
    prep = make_feature_pipeline(
        scaled=spec.linear,
        status_as_category=spec.linear and config.STATUS_AS_CATEGORY_LINEAR,
    )
    return Pipeline([("prep", prep), ("model", model if model is not None else spec.factory())])

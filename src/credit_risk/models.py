"""Catálogo de modelos e montagem do pipeline completo de cada um.

Cada modelo recebe o pré-processamento adequado: modelos lineares com escala
e ``pay_status`` como categoria; árvores com os números como estão.
Hiperparâmetros aqui são padrões razoáveis, sem tuning (Etapa 11).
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from credit_risk import config
from credit_risk.preprocessing import make_feature_pipeline


def _load_best_params() -> dict:
    """Hiperparâmetros adotados no tuning (vazio se ainda não rodou)."""
    if not config.BEST_PARAMS.exists():
        return {}
    return json.loads(config.BEST_PARAMS.read_text())


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


def build_pipeline(
    name: str,
    model: BaseEstimator | None = None,
    tuned: bool = False,
    exclude: tuple[str, ...] = (),
) -> Pipeline:
    """Pipeline completo (features → pré-processamento → modelo) de um modelo do catálogo.

    Args:
        name: Chave em ``CATALOG``.
        model: Estimador a usar no lugar do padrão.
        tuned: Usa os hiperparâmetros adotados na Etapa 11 (``best_params.json``),
            se houver. O padrão é False para manter reproduzíveis os experimentos
            das Etapas 9 e 10.
        exclude: Colunas a deixar fora do modelo (ex.: atributos sensíveis).

    Returns:
        Pipeline não ajustado.
    """
    spec = CATALOG[name]
    if model is None and tuned:
        params = _load_best_params().get(name)
        if params:
            model = spec.factory().set_params(**params)
    prep = make_feature_pipeline(
        scaled=spec.linear,
        status_as_category=spec.linear and config.STATUS_AS_CATEGORY_LINEAR,
        exclude=tuple(exclude),
    )
    return Pipeline([("prep", prep), ("model", model if model is not None else spec.factory())])


ENSEMBLES: dict[str, tuple[str, list[str]]] = {
    "voting_xgb_rf": ("soft", ["xgboost", "random_forest"]),
    "voting_xgb_rf_logit": ("soft", ["xgboost", "random_forest", "logistic"]),
    "stacking_xgb_rf_logit": ("stacking", ["xgboost", "random_forest", "logistic"]),
}


def build_ensemble(name: str) -> VotingClassifier | StackingClassifier:
    """Combinação de modelos ajustados, cada um com o próprio pré-processamento.

    Na votação soft, a probabilidade final é a média das probabilidades. No
    stacking, uma regressão logística aprende o peso de cada modelo a partir de
    previsões out-of-fold geradas **dentro** do treino (sem vazamento).

    Args:
        name: Chave em ``ENSEMBLES``.

    Returns:
        Ensemble não ajustado.
    """
    kind, members = ENSEMBLES[name]
    estimators = [(m, build_pipeline(m, tuned=True)) for m in members]
    if kind == "soft":
        return VotingClassifier(estimators, voting="soft")
    return StackingClassifier(
        estimators,
        final_estimator=LogisticRegression(max_iter=2_000),
        cv=StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.SEED),
        stack_method="predict_proba",
    )


def final_pipeline() -> Pipeline:
    """O modelo oficial do projeto: única fonte de verdade para decisão, API e dashboard.

    XGBoost com os hiperparâmetros da Etapa 11, sem os atributos sensíveis
    removidos na Etapa 13A (sexo, estado civil e idade).

    Returns:
        Pipeline não ajustado.
    """
    return build_pipeline(config.FINAL_MODEL, tuned=True, exclude=tuple(config.MODEL_EXCLUDE))

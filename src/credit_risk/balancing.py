"""Balanceamento de classes: nenhum × pesos por classe × SMOTE.

Os dois métodos fazem o modelo "enxergar" um mundo com 50% de caloteiros, o
que tende a inflar as probabilidades. Tudo é aplicado só no ``fit`` de cada
fold: os pesos são calculados com o treino do fold, e o SMOTE nunca toca os
dados de avaliação.
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from credit_risk import config
from credit_risk.models import CATALOG, build_pipeline

STRATEGIES = ["none", "class_weight", "smote"]


class BalancedWeights(ClassifierMixin, BaseEstimator):
    """Envelope que treina o estimador com pesos que igualam as classes.

    Os pesos são calculados no ``fit``, com o alvo recebido (o treino do fold).
    Com 22% de inadimplentes, cada um pesa cerca de 3,5 vezes um bom pagador.

    Args:
        estimator: Classificador que aceita ``sample_weight`` no ``fit``.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: object, y: np.ndarray) -> "BalancedWeights":
        """Ajusta o estimador com pesos balanceados.

        Args:
            X: Features.
            y: Alvo binário.

        Returns:
            O próprio objeto ajustado.
        """
        weights = compute_sample_weight("balanced", y)
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=weights)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        """Probabilidades do estimador ajustado.

        Args:
            X: Features.

        Returns:
            Matriz ``(n, 2)`` de probabilidades.
        """
        return self.estimator_.predict_proba(X)

    def predict(self, X: object) -> np.ndarray:
        """Classe prevista com corte em 50%.

        Args:
            X: Features.

        Returns:
            Vetor de classes.
        """
        return self.estimator_.predict(X)


def balanced_pipeline(name: str, strategy: str) -> Pipeline | ImbPipeline:
    """Pipeline de um modelo do catálogo com a estratégia de balanceamento.

    Args:
        name: Chave em ``models.CATALOG``.
        strategy: ``"none"``, ``"class_weight"`` ou ``"smote"``.

    Returns:
        Pipeline não ajustado.

    Raises:
        ValueError: Se a estratégia for desconhecida.
    """
    if strategy == "none":
        return build_pipeline(name)
    if strategy == "class_weight":
        return build_pipeline(name, BalancedWeights(CATALOG[name].factory()))
    if strategy == "smote":
        base = build_pipeline(name)
        feature_steps = base.named_steps["prep"].steps
        return ImbPipeline(
            [
                *feature_steps,
                ("smote", SMOTE(random_state=config.SEED)),
                ("model", base.named_steps["model"]),
            ]
        )
    raise ValueError(f"Estratégia desconhecida: {strategy}")

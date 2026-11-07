"""Calibração das probabilidades e política de decisão por custo.

Parte A (calibração): o XGBoost ajustado já sai quase calibrado (Etapa 10);
aqui se testa se uma correção sigmoid ou isotônica melhora o Brier.
"""

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

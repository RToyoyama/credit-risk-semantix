"""Garante que dados, modelo e previsões existam, reconstruindo só o que faltar.

Usado pelo dashboard (e útil no Docker): num ambiente novo, como o Streamlit
Community Cloud, baixa os dados da UCI, refaz limpeza e split (com os IDs do
holdout versionados), treina o modelo oficial e gera as previsões. Nada disso
reavalia o holdout: métricas e decisões já estão em ``reports/`` (Etapa 13).
"""

import logging
from dataclasses import dataclass

import joblib
import pandas as pd

from credit_risk import config
from credit_risk.cleaning import build_interim
from credit_risk.data import build_raw
from credit_risk.decision import exposure
from credit_risk.evaluate import oof_predictions
from credit_risk.models import final_pipeline
from credit_risk.segmentation import SegmentEncoder
from credit_risk.split import build_split, load_train

logger = logging.getLogger(__name__)


@dataclass
class Artifacts:
    """Tudo o que o dashboard precisa.

    Attributes:
        model: Pipeline final ajustado no treino completo.
        train: Dados limpos de treino (com o alvo).
        holdout: Dados limpos do holdout (com o alvo).
        oof: Previsões fora do fold do treino (id, alvo, ead, pd).
        holdout_pred: Previsões do holdout (id, alvo, ead, pd, segmento).
    """

    model: object
    train: pd.DataFrame
    holdout: pd.DataFrame
    oof: pd.DataFrame
    holdout_pred: pd.DataFrame


def ensure_artifacts() -> Artifacts:
    """Carrega os artefatos, reconstruindo os que estiverem faltando.

    Returns:
        Artefatos prontos para uso.
    """
    if not config.RAW_PARQUET.exists():
        logger.info("Baixando os dados da UCI...")
        build_raw()
    if not config.INTERIM_PARQUET.exists():
        build_interim()
    if not (config.TRAIN_PARQUET.exists() and config.HOLDOUT_PARQUET.exists()):
        build_split()
    train = load_train()
    holdout = pd.read_parquet(config.HOLDOUT_PARQUET)
    X, y = train.drop(columns=[config.TARGET]), train[config.TARGET]

    if config.FINAL_MODEL_PATH.exists():
        model = joblib.load(config.FINAL_MODEL_PATH)
    else:
        logger.info("Treinando o modelo final...")
        model = final_pipeline().fit(X, y)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, config.FINAL_MODEL_PATH)

    if config.OOF_PREDICTIONS.exists():
        oof = pd.read_parquet(config.OOF_PREDICTIONS)
    else:
        logger.info("Gerando previsões fora do fold...")
        oof = pd.DataFrame(
            {
                "id": train["id"],
                config.TARGET: y,
                "ead": exposure(train),
                "pd": oof_predictions(final_pipeline(), X, y),
            }
        )
        oof.to_parquet(config.OOF_PREDICTIONS, index=False)

    if config.HOLDOUT_PREDICTIONS.exists():
        holdout_pred = pd.read_parquet(config.HOLDOUT_PREDICTIONS)
    else:
        holdout_pred = pd.DataFrame(
            {
                "id": holdout["id"],
                config.TARGET: holdout[config.TARGET],
                "ead": exposure(holdout),
                "pd": model.predict_proba(holdout.drop(columns=[config.TARGET]))[:, 1],
                "segment": SegmentEncoder().fit(train).predict(holdout),
            }
        )
        holdout_pred.to_parquet(config.HOLDOUT_PREDICTIONS, index=False)

    return Artifacts(model, train, holdout, oof, holdout_pred)

"""Segmentação comportamental: padronização + PCA (+ K-means na Parte B).

Tudo que aprende com os dados (médias, desvios, direções do PCA) fica
dentro de um ``Pipeline``, para poder ser ajustado só no treino de cada fold.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from credit_risk import config
from credit_risk.features import behavior_features


def make_pca_pipeline(n_components: int | float | None = None) -> Pipeline:
    """Monta o pipeline dados limpos → features comportamentais → escala → PCA.

    Args:
        n_components: Componentes a manter (None mantém todos).

    Returns:
        Pipeline não ajustado.
    """
    return Pipeline(
        [
            ("behavior", FunctionTransformer(behavior_features)),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=config.SEED)),
        ]
    )


def explained_variance_table(pca: PCA) -> pd.DataFrame:
    """Variância explicada por componente, acumulada e autovalor.

    Args:
        pca: PCA já ajustado sobre dados padronizados.

    Returns:
        Tabela indexada por ``PC1``, ``PC2``...
    """
    ratio = pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "variancia": ratio,
            "acumulada": np.cumsum(ratio),
            "autovalor": pca.explained_variance_,
        },
        index=[f"PC{i}" for i in range(1, len(ratio) + 1)],
    )


def choose_n_components(cumulative: pd.Series, target: float = config.PCA_VARIANCE_TARGET) -> int:
    """Menor número de componentes cuja variância acumulada atinge a meta.

    Args:
        cumulative: Variância explicada acumulada (crescente).
        target: Meta, entre 0 e 1.

    Returns:
        Número de componentes.
    """
    return int(np.searchsorted(cumulative.to_numpy(), target - 1e-12) + 1)


def loadings(pca: PCA, feature_names: list[str], n_components: int | None = None) -> pd.DataFrame:
    """Cargas: quanto cada feature original pesa em cada componente.

    Args:
        pca: PCA ajustado.
        feature_names: Nomes das features de entrada, na ordem.
        n_components: Quantos componentes mostrar (None = todos).

    Returns:
        Tabela features × componentes.
    """
    comps = pca.components_[:n_components]
    return pd.DataFrame(
        comps.T, index=feature_names, columns=[f"PC{i}" for i in range(1, len(comps) + 1)]
    )

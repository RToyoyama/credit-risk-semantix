"""Pré-processamento: o "tradutor" entre as features e cada tipo de modelo.

Tudo o que aprende com os dados (médias, desvios, lista de categorias) fica
dentro do ``ColumnTransformer``, para ser reajustado só no treino de cada fold.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_risk import config
from credit_risk.features import build_features


def feature_columns(
    engineered: bool = True, status_as_category: bool = False
) -> tuple[list[str], list[str]]:
    """Separa as colunas em numéricas e categóricas.

    Args:
        engineered: Inclui as features de resumo da Etapa 7.
        status_as_category: Trata ``pay_status_*`` como categoria em vez de número.

    Returns:
        Tupla ``(numéricas, categóricas)``.
    """
    numeric = [
        c for c in config.RAW_NUMERIC_COLS if not (status_as_category and c in config.PAY_STATUS)
    ]
    if engineered:
        numeric += config.ENGINEERED_FEATURES
    categorical = config.CATEGORICAL_COLS + (config.PAY_STATUS if status_as_category else [])
    return numeric, categorical


def make_preprocessor(
    scaled: bool = True, engineered: bool = True, status_as_category: bool = False
) -> ColumnTransformer:
    """Escala (opcional) nos números e one-hot nas categorias, com saída pandas.

    Args:
        scaled: Padroniza os números (modelos lineares). Árvores não precisam.
        engineered: Inclui as features de resumo.
        status_as_category: Trata ``pay_status_*`` como categoria.

    Returns:
        ``ColumnTransformer`` não ajustado; colunas fora das listas são descartadas.
    """
    numeric, categorical = feature_columns(engineered, status_as_category)
    onehot = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=config.MIN_CATEGORY_COUNT,
        sparse_output=False,
        dtype=float,
    )
    return ColumnTransformer(
        [
            ("num", StandardScaler() if scaled else "passthrough", numeric),
            ("cat", onehot, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_feature_pipeline(
    scaled: bool = True, engineered: bool = True, status_as_category: bool = False
) -> Pipeline:
    """Dados limpos → features determinísticas → pré-processamento.

    Args:
        scaled: Padroniza os números.
        engineered: Inclui as features de resumo.
        status_as_category: Trata ``pay_status_*`` como categoria.

    Returns:
        Pipeline sem modelo; a Etapa 9 acrescenta o classificador no final.
    """
    return Pipeline(
        [
            ("features", FunctionTransformer(build_features)),
            ("prep", make_preprocessor(scaled, engineered, status_as_category)),
        ]
    )

"""Holdout estratificado e reprodutível, "lacrado" por uma lista versionada de IDs.

O holdout é definido **uma única vez** (20%, estratificado pelo alvo, seed fixa)
e seus IDs são salvos em ``reports/holdout_ids.csv``, que vai para o Git. A
partir daí, a lista é a fonte de verdade: mesmo que uma versão futura do
scikit-learn sorteie diferente com a mesma seed, o holdout não muda.

Uso: ``python -m credit_risk.split``.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk import config
from credit_risk.cleaning import load_interim

logger = logging.getLogger(__name__)


def make_holdout_ids(
    df: pd.DataFrame,
    test_size: float = config.HOLDOUT_SIZE,
    seed: int = config.SEED,
) -> pd.Series:
    """Sorteia os IDs do holdout, estratificando pelo alvo.

    Args:
        df: Dados limpos com as colunas ``id`` e do alvo.
        test_size: Fração separada para o holdout.
        seed: Semente do sorteio.

    Returns:
        IDs do holdout, ordenados.
    """
    _, holdout = train_test_split(
        df[["id", config.TARGET]],
        test_size=test_size,
        stratify=df[config.TARGET],
        random_state=seed,
    )
    return holdout["id"].sort_values().reset_index(drop=True)


def split_by_ids(df: pd.DataFrame, holdout_ids: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa treino e holdout a partir de uma lista de IDs.

    Args:
        df: Dados limpos.
        holdout_ids: IDs que formam o holdout.

    Returns:
        Tupla ``(treino, holdout)``.

    Raises:
        ValueError: Se algum ID da lista não existir nos dados.
    """
    missing = set(holdout_ids) - set(df["id"])
    if missing:
        raise ValueError(f"{len(missing)} IDs do holdout não existem nos dados.")
    is_holdout = df["id"].isin(holdout_ids)
    return df[~is_holdout].reset_index(drop=True), df[is_holdout].reset_index(drop=True)


def get_holdout_ids(df: pd.DataFrame, ids_path: Path = config.HOLDOUT_IDS) -> pd.Series:
    """Lê a lista lacrada de IDs; se ainda não existir, sorteia e salva.

    Args:
        df: Dados limpos (usados só se a lista ainda não existir).
        ids_path: Arquivo CSV com a coluna ``id``.

    Returns:
        IDs do holdout.
    """
    if ids_path.exists():
        logger.info("Usando holdout lacrado: %s", ids_path)
        return pd.read_csv(ids_path)["id"]
    ids = make_holdout_ids(df)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids.to_frame().to_csv(ids_path, index=False)
    logger.info("Holdout criado e lacrado: %s (%d IDs)", ids_path, len(ids))
    return ids


def build_split() -> tuple[Path, Path]:
    """Gera ``train.parquet`` e ``holdout.parquet`` a partir da camada interim.

    Returns:
        Caminhos do treino e do holdout.
    """
    df = load_interim()
    train, holdout = split_by_ids(df, get_holdout_ids(df))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(config.TRAIN_PARQUET, index=False)
    holdout.to_parquet(config.HOLDOUT_PARQUET, index=False)
    for name, part in (("treino", train), ("holdout", holdout)):
        logger.info("%-7s %6d linhas | default %.4f", name, len(part), part[config.TARGET].mean())
    return config.TRAIN_PARQUET, config.HOLDOUT_PARQUET


def load_train() -> pd.DataFrame:
    """Lê o conjunto de treino (o único usado até a Etapa 13).

    Returns:
        Dados de treino.

    Raises:
        FileNotFoundError: Se ``make split`` ainda não foi executado.
    """
    if not config.TRAIN_PARQUET.exists():
        raise FileNotFoundError(f"{config.TRAIN_PARQUET} não existe. Rode `make split`.")
    return pd.read_parquet(config.TRAIN_PARQUET)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    build_split()

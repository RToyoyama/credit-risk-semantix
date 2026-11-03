"""Limpeza determinística: renomeação, categorias não documentadas e flags.

Todas as regras são fixas (não aprendem nada dos dados), por isso podem ser
aplicadas antes da separação do holdout sem causar vazamento de dados.

Uso: ``python -m credit_risk.cleaning``.
"""

import logging
from pathlib import Path

import pandas as pd

from credit_risk import config
from credit_risk.data import load_raw

logger = logging.getLogger(__name__)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia para snake_case e alinha os meses (índice 1 = mais recente).

    Na fonte, o status de setembro é ``PAY_0``, mas a fatura de setembro é
    ``BILL_AMT1``. Após a renomeação, ``pay_status_1``, ``bill_amt_1`` e
    ``pay_amt_1`` referem-se todos a setembro/2005.

    Args:
        df: Dados com os nomes originais da UCI.

    Returns:
        Cópia com as colunas renomeadas, na ordem do schema limpo.
    """
    return df.rename(columns=config.RENAME_MAP)[list(config.RENAME_MAP.values())]


def group_undocumented_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa valores não documentados na categoria "outros" e marca a origem.

    ``education`` 0/5/6 vira 4 e ``marriage`` 0 vira 3. As flags
    ``education_undocumented`` e ``marriage_undocumented`` guardam a
    informação de que o valor original não era documentado.

    Args:
        df: Dados já renomeados.

    Returns:
        Cópia com as categorias agrupadas e as duas flags.
    """
    out = df.copy()
    edu_bad = ~out["education"].isin(config.DOCUMENTED_CATEGORIES["EDUCATION"])
    mar_bad = ~out["marriage"].isin(config.DOCUMENTED_CATEGORIES["MARRIAGE"])
    out["education_undocumented"] = edu_bad.astype("int8")
    out["marriage_undocumented"] = mar_bad.astype("int8")
    out.loc[edu_bad, "education"] = config.EDUCATION_OTHERS
    out.loc[mar_bad, "marriage"] = config.MARRIAGE_OTHERS
    return out


def count_status_bill_mismatch(df: pd.DataFrame) -> pd.Series:
    """Conta, por cliente, os meses em que o status contradiz a fatura.

    Contradição = status "sem consumo" (-2) com fatura positiva, ou status
    "em atraso" (>= 1) com fatura zero ou negativa. Não é tratado como erro
    (são frequentes demais); serve de sinal de qualidade a investigar na EDA.

    Args:
        df: Dados já renomeados.

    Returns:
        Série inteira de 0 a 6, alinhada ao índice de ``df``.
    """
    status = df[config.PAY_STATUS].to_numpy()
    bill = df[config.BILL_AMT].to_numpy()
    mismatch = ((status == -2) & (bill > 0)) | ((status >= 1) & (bill <= 0))
    return pd.Series(mismatch.sum(axis=1).astype("int8"), index=df.index)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica toda a limpeza determinística.

    Args:
        df: Dados raw com os nomes originais da UCI.

    Returns:
        Dados limpos, com o mesmo número de linhas e os mesmos IDs.
    """
    out = group_undocumented_categories(rename_columns(df))
    out["n_status_bill_mismatch"] = count_status_bill_mismatch(out)
    return out


def build_interim(path: Path = config.INTERIM_PARQUET) -> Path:
    """Lê a camada raw, limpa e salva a camada interim.

    Args:
        path: Destino do Parquet limpo.

    Returns:
        O caminho salvo.
    """
    df = clean(load_raw())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Interim salvo: %s (%d linhas, %d colunas)", path, *df.shape)
    return path


def load_interim() -> pd.DataFrame:
    """Lê a camada interim.

    Returns:
        Dados limpos.

    Raises:
        FileNotFoundError: Se ``make split`` ainda não foi executado.
    """
    if not config.INTERIM_PARQUET.exists():
        raise FileNotFoundError(f"{config.INTERIM_PARQUET} não existe. Rode `make split`.")
    return pd.read_parquet(config.INTERIM_PARQUET)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    build_interim()

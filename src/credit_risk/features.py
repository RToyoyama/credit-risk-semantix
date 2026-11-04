"""Features determinísticas: regras fixas, calculadas linha a linha.

Nenhuma função aqui aprende com os dados (não usa médias, quantis nem o
alvo), por isso podem ser aplicadas a qualquer conjunto sem vazamento.
"""

import numpy as np
import pandas as pd

from credit_risk import config


def utilization(df: pd.DataFrame) -> pd.DataFrame:
    """Fração do limite usada em cada mês (fatura ÷ limite).

    Fatura negativa (crédito a favor) vira 0; valores acima de 150% são
    raros e são limitados para não distorcer a escala.

    Args:
        df: Dados limpos com ``limit_bal`` e ``bill_amt_1..6``.

    Returns:
        Colunas ``utilization_1..6``.
    """
    ratio = df[config.BILL_AMT].to_numpy() / df[["limit_bal"]].to_numpy()
    low, high = config.UTILIZATION_CLIP
    return pd.DataFrame(np.clip(ratio, low, high), columns=config.UTILIZATION, index=df.index)


def payment_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Fração da fatura anterior que o cliente pagou em cada mês.

    O pagamento do mês ``i`` quita a fatura do mês ``i + 1`` (o anterior).
    Sem fatura a pagar (zero ou negativa), considera-se em dia (1). Pagar
    mais que a fatura conta como quitar (1).

    Args:
        df: Dados limpos com ``pay_amt_1..6`` e ``bill_amt_1..6``.

    Returns:
        Colunas ``payment_ratio_1..5``.
    """
    paid = df[config.PAY_AMT[:-1]].to_numpy(dtype=float)
    owed = df[config.BILL_AMT[1:]].to_numpy(dtype=float)
    ratio = np.divide(paid, owed, out=np.ones_like(paid), where=owed > 0)
    low, high = config.PAYMENT_RATIO_CLIP
    return pd.DataFrame(np.clip(ratio, low, high), columns=config.PAYMENT_RATIO, index=df.index)


def delay_months(df: pd.DataFrame) -> pd.DataFrame:
    """Meses de atraso em cada mês (status <= 0 significa sem atraso).

    Args:
        df: Dados limpos com ``pay_status_1..6``.

    Returns:
        Colunas ``delay_1..6``.
    """
    delay = df[config.PAY_STATUS].clip(lower=0).to_numpy()
    return pd.DataFrame(delay, columns=config.DELAY, index=df.index)


def behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz comportamental usada na segmentação (sem alvo e sem perfil).

    Args:
        df: Dados limpos.

    Returns:
        17 colunas: utilização (6), taxa de pagamento (5) e atraso (6).
    """
    return pd.concat([utilization(df), payment_ratio(df), delay_months(df)], axis=1)

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


def _recent_minus_old(values: pd.DataFrame) -> pd.Series:
    """Média dos meses recentes menos a média dos antigos (positivo = subindo)."""
    k = config.RECENT_MONTHS
    return values.iloc[:, :k].mean(axis=1) - values.iloc[:, k:].mean(axis=1)


def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de resumo do histórico, guiadas pela EDA e pela segmentação.

    Todas são regras fixas calculadas linha a linha (sem aprender dos dados).

    Args:
        df: Dados limpos.

    Returns:
        Colunas listadas em ``config.ENGINEERED_FEATURES``.
    """
    util = utilization(df)
    ratio = payment_ratio(df)
    delay = delay_months(df)
    status = df[config.PAY_STATUS]
    out = pd.DataFrame(
        {
            "log_limit": np.log(df["limit_bal"]),
            "util_mean": util.mean(axis=1),
            "util_recent": util["utilization_1"],
            "util_trend": _recent_minus_old(util),
            "util_volatility": util.std(axis=1),
            "pay_ratio_mean": ratio.mean(axis=1),
            "pay_ratio_recent": ratio["payment_ratio_1"],
            "paid_full_all": (ratio >= 1).all(axis=1).astype("int8"),
            "delay_max": delay.max(axis=1),
            "delay_months": (delay > 0).sum(axis=1),
            "delay_recent": delay["delay_1"],
            "delay_trend": _recent_minus_old(delay),
            "revolving_months": (status == 0).sum(axis=1),
        },
        index=df.index,
    )
    return out[config.ENGINEERED_FEATURES]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas originais (sem ID e sem alvo) + features de resumo.

    As colunas que o modelo final não usa (``config.MODEL_EXCLUDE``: sexo,
    estado civil e idade) podem estar ausentes: a API não as coleta (LGPD,
    princípio da necessidade). Qualquer outra coluna ausente gera erro.

    Args:
        df: Dados limpos.

    Returns:
        Matriz de features determinísticas, pronta para o pré-processamento.
    """
    wanted = config.CATEGORICAL_COLS + config.RAW_NUMERIC_COLS
    present = [c for c in wanted if c in df.columns or c not in config.MODEL_EXCLUDE]
    base = df[present]
    return pd.concat([base, engineered_features(df)], axis=1)

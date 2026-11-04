"""Interpretabilidade: colinearidade (VIF) e razões de chances da logística.

Os coeficientes da logística completa não podem ser lidos um a um, porque
várias variáveis carregam a mesma informação (multicolinearidade). Para
interpretar, usa-se uma logística enxuta, com variáveis pouco correlacionadas,
ajustada com ``statsmodels`` para obter intervalos de confiança.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from credit_risk import config


def vif_table(X: pd.DataFrame) -> pd.Series:
    """Fator de inflação da variância de cada coluna.

    VIF = 1 / (1 − R²), em que R² mede o quanto aquela coluna é explicada
    pelas demais. VIF acima de ``config.VIF_ALERT`` indica colinearidade.

    Args:
        X: Variáveis numéricas.

    Returns:
        VIF por coluna, em ordem decrescente.
    """
    Xc = sm.add_constant(X.astype(float))
    values = [variance_inflation_factor(Xc.to_numpy(), i) for i in range(1, Xc.shape[1])]
    return pd.Series(values, index=X.columns, name="vif").sort_values(ascending=False)


def odds_ratio_table(X: pd.DataFrame, y: pd.Series, level: float = config.CI_LEVEL) -> pd.DataFrame:
    """Logística sem regularização com razões de chances e intervalos de confiança.

    Razão de chances = exp(coeficiente): quanto a chance de default é
    multiplicada quando a variável aumenta 1 unidade, mantidas as demais.

    Args:
        X: Variáveis explicativas (em unidades naturais).
        y: Alvo binário.
        level: Nível de confiança.

    Returns:
        Tabela com razão de chances, IC, p-valor e pseudo-R² (atributo ``attrs``).
    """
    model = sm.Logit(y.astype(float), sm.add_constant(X.astype(float))).fit(disp=0)
    ci = np.exp(model.conf_int(alpha=1 - level))
    table = pd.DataFrame(
        {
            "razao_chances": np.exp(model.params),
            "ic_inf": ci[0],
            "ic_sup": ci[1],
            "p_valor": model.pvalues,
        }
    ).drop(index="const")
    table.attrs["pseudo_r2"] = float(model.prsquared)
    return table

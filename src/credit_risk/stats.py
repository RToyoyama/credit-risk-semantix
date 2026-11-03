"""Inferência estatística: intervalos de confiança, testes e tamanhos de efeito.

Com amostras grandes, quase toda diferença é "significativa". Por isso todo
teste aqui devolve também um tamanho de efeito, que diz se a diferença é
grande o bastante para importar.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.stats.contingency import association
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from credit_risk import config


@dataclass(frozen=True)
class TestResult:
    """Resultado de um teste de hipótese com tamanho de efeito.

    Attributes:
        variable: Variável testada.
        test: Nome do teste.
        statistic: Estatística do teste.
        p_value: p-valor bruto (antes da correção de Holm).
        effect_name: Nome da medida de efeito.
        effect: Valor do efeito.
        n: Tamanho da amostra.
    """

    variable: str
    test: str
    statistic: float
    p_value: float
    effect_name: str
    effect: float
    n: int


def bootstrap_ci(
    values: np.ndarray | pd.Series,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = config.N_BOOT,
    level: float = config.CI_LEVEL,
    seed: int = config.SEED,
) -> tuple[float, float, float]:
    """Intervalo de confiança por bootstrap (método dos percentis).

    Args:
        values: Amostra observada.
        statistic: Função aplicada a cada reamostragem.
        n_boot: Número de reamostragens.
        level: Nível de confiança (ex.: 0.95).
        seed: Semente do sorteio.

    Returns:
        Tupla ``(estimativa, limite_inferior, limite_superior)``.
    """
    arr = np.asarray(values)
    rng = np.random.default_rng(seed)
    boots = np.array([statistic(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n_boot)])
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(boots, [tail, 100 - tail])
    return float(statistic(arr)), float(low), float(high)


def chi2_cramers_v(x: pd.Series, y: pd.Series) -> TestResult:
    """Teste qui-quadrado de independência com V de Cramér.

    Usa a tabela sem correção de Yates, para que o V seja coerente com a
    estatística reportada.

    Args:
        x: Variável categórica.
        y: Alvo binário.

    Returns:
        Resultado com o V de Cramér (0 = sem relação, 1 = relação perfeita).
    """
    table = pd.crosstab(x, y)
    chi2, p, _, _ = sps.chi2_contingency(table, correction=False)
    v = association(table.to_numpy(), method="cramer", correction=False)
    return TestResult(
        str(x.name), "qui-quadrado", float(chi2), float(p), "Cramér V", float(v), len(x)
    )


def mann_whitney_rank_biserial(values: pd.Series, y: pd.Series) -> TestResult:
    """Teste de Mann-Whitney com correlação rank-biserial.

    O efeito é positivo quando os inadimplentes (``y == 1``) tendem a ter
    valores maiores. Interpretação: ``(r + 1) / 2`` é a chance de um
    inadimplente sorteado ter valor maior que um adimplente sorteado.

    Args:
        values: Variável numérica.
        y: Alvo binário.

    Returns:
        Resultado com a rank-biserial (-1 a 1).
    """
    pos = values[y == 1].to_numpy()
    neg = values[y == 0].to_numpy()
    u, p = sps.mannwhitneyu(pos, neg, alternative="two-sided")
    r = 2 * u / (len(pos) * len(neg)) - 1
    return TestResult(
        str(values.name), "Mann-Whitney", float(u), float(p), "rank-biserial", float(r), len(values)
    )


def effect_magnitude(effect: float) -> str:
    """Classifica o tamanho de efeito pela convenção de Cohen.

    Args:
        effect: Cramér V ou rank-biserial (o sinal é ignorado).

    Returns:
        "desprezível", "pequeno", "médio" ou "grande".
    """
    label = "desprezível"
    for name, threshold in config.EFFECT_THRESHOLDS.items():
        if abs(effect) >= threshold:
            label = name
    return label


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Ajusta p-valores para múltiplas comparações (Holm-Bonferroni).

    Args:
        p_values: p-valores brutos.

    Returns:
        p-valores ajustados, na mesma ordem.
    """
    return multipletests(p_values, method="holm")[1]


def results_table(results: list[TestResult], alpha: float = config.ALPHA) -> pd.DataFrame:
    """Consolida testes: p ajustado por Holm, magnitude e relevância.

    Uma variável é relevante se for significativa após Holm **e** tiver
    efeito pelo menos pequeno.

    Args:
        results: Testes já executados.
        alpha: Nível de significância.

    Returns:
        Tabela ordenada pelo tamanho de efeito absoluto.
    """
    df = pd.DataFrame([r.__dict__ for r in results])
    df["p_holm"] = holm_adjust(df["p_value"])
    df["significativo"] = df["p_holm"] < alpha
    df["magnitude"] = df["effect"].map(effect_magnitude)
    df["relevante"] = df["significativo"] & (df["magnitude"] != "desprezível")
    return df.sort_values("effect", key=np.abs, ascending=False).reset_index(drop=True)


def rate_by_group(
    df: pd.DataFrame, col: str, target: str = config.TARGET, level: float = config.CI_LEVEL
) -> pd.DataFrame:
    """Taxa de default por grupo com intervalo de Wilson.

    Args:
        df: Dados.
        col: Coluna que define os grupos.
        target: Alvo binário.
        level: Nível de confiança.

    Returns:
        Tabela com n, defaults, taxa e IC por grupo.
    """
    g = df.groupby(col, observed=True)[target].agg(n="size", defaults="sum").reset_index()
    g["taxa"] = g["defaults"] / g["n"]
    low, high = proportion_confint(g["defaults"], g["n"], alpha=1 - level, method="wilson")
    g["ic_inf"], g["ic_sup"] = low, high
    return g

"""Estilo visual e gráficos reutilizáveis (matplotlib).

Regra de cor: cinza para contexto, um único azul de destaque para o que a
frase do título afirma. A taxa média aparece sempre como linha de referência.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from credit_risk import config

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
CONTEXT = "#b9b8b2"
HIGHLIGHT = "#2a78d6"
ALERT = "#e34948"


def pct(value: float, decimals: int = 1) -> str:
    """Formata uma proporção como porcentagem no padrão brasileiro.

    Args:
        value: Proporção (ex.: 0.2212).
        decimals: Casas decimais.

    Returns:
        Texto como "22,1%".
    """
    return f"{value:.{decimals}%}".replace(".", ",")


def set_style() -> None:
    """Aplica o estilo padrão do projeto a todos os gráficos."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.constrained_layout.use": True,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "legend.frameon": False,
        }
    )


def plot_rate_by_group(
    rates: pd.DataFrame,
    col: str,
    title: str,
    overall: float,
    highlight: list | None = None,
    labels: dict | None = None,
    xlabel: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Barras da taxa de default por grupo, com IC e linha da média geral.

    Args:
        rates: Saída de ``stats.rate_by_group`` (colunas ``taxa``, ``ic_inf``, ``ic_sup``, ``n``).
        col: Coluna dos grupos.
        title: Título que afirma o insight.
        overall: Taxa geral, desenhada como referência.
        highlight: Grupos a destacar em azul (o resto fica cinza).
        labels: Rótulos legíveis para os grupos.
        xlabel: Rótulo do eixo x.
        ax: Eixo existente; se None, cria um novo.

    Returns:
        O eixo desenhado.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.6))
    highlight = highlight or []
    names = [str((labels or {}).get(g, g)) for g in rates[col]]
    colors = [HIGHLIGHT if g in highlight else CONTEXT for g in rates[col]]
    x = range(len(rates))
    ax.bar(x, rates["taxa"], color=colors, width=0.62, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(
        x,
        rates["taxa"],
        yerr=[rates["taxa"] - rates["ic_inf"], rates["ic_sup"] - rates["taxa"]],
        fmt="none",
        ecolor=INK_2,
        elinewidth=1,
        capsize=3,
    )
    ax.axhline(overall, color=INK_2, linestyle="--", linewidth=1, zorder=4)
    ax.annotate(
        f"média\n{pct(overall)}",
        xy=(1, overall),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        color=INK_2,
        fontsize=8,
        annotation_clip=False,
    )
    for xi, (rate, top) in enumerate(zip(rates["taxa"], rates["ic_sup"], strict=True)):
        ax.annotate(
            pct(rate, 0),
            xy=(xi, top),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
            zorder=5,
            bbox={"boxstyle": "square,pad=0.1", "facecolor": SURFACE, "edgecolor": "none"},
        )
    names = [
        f"{name}\nn={n:,}".replace(",", ".") for name, n in zip(names, rates["n"], strict=True)
    ]
    ax.margins(y=0.12)
    ax.set_xticks(list(x), names, fontsize=8 if len(names) > 6 else None)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("taxa de default")
    ax.set_xlabel(xlabel or "")
    ax.set_title(title)
    return ax


def save(fig: Figure, name: str, folder: Path = config.FIGURES_DIR) -> Path:
    """Salva a figura em ``reports/figures``.

    Args:
        fig: Figura do matplotlib.
        name: Nome do arquivo, sem extensão.
        folder: Pasta de destino.

    Returns:
        Caminho do PNG salvo.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path

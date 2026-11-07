"""Estilo visual do dashboard (as mesmas cores do resto do projeto) e formatação pt-BR."""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from credit_risk.viz import CATEGORICAL, CONTEXT, GRID, HIGHLIGHT, INK, INK_2, SURFACE

ALERT = "#e34948"

pio.templates["credito"] = go.layout.Template(
    layout={
        "font": {"family": "sans-serif", "size": 13, "color": INK},
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "colorway": CATEGORICAL,
        "xaxis": {"gridcolor": GRID, "linecolor": GRID, "zeroline": False},
        "yaxis": {"gridcolor": GRID, "linecolor": GRID, "zeroline": False},
        "margin": {"l": 10, "r": 10, "t": 50, "b": 10},
        "separators": ",.",
        "hoverlabel": {"bgcolor": SURFACE, "font_color": INK},
    }
)
pio.templates.default = "credito"

__all__ = ["ALERT", "CATEGORICAL", "CONTEXT", "HIGHLIGHT", "INK_2", "insight", "md", "money", "pct"]


def _br(text: str) -> str:
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def money(value: float, unit: str = "mi") -> str:
    """Formata NT$ em milhões, em mil ou em reais inteiros (``unit="auto"``), padrão brasileiro."""
    if unit == "auto":
        if abs(value) >= 1e6:
            unit = "mi"
        elif abs(value) >= 1e4:
            unit = "mil"
        else:
            return _br(f"NT$ {value:,.0f}")
    if unit == "mi":
        return _br(f"NT$ {value / 1e6:,.1f} mi")
    return _br(f"NT$ {value / 1e3:,.1f} mil")


def pct(value: float, decimals: int = 1) -> str:
    """Formata proporção como porcentagem pt-BR."""
    return f"{value:.{decimals}%}".replace(".", ",")


def md(text: str) -> str:
    """Escapa o cifrão: no Markdown do Streamlit, texto entre dois $ vira fórmula (LaTeX)."""
    return text.replace("$", "\\$")


def insight(title: str, subtitle: str | None = None) -> None:
    """Título de seção que afirma o insight, com uma linha de contexto opcional."""
    st.subheader(md(title))
    if subtitle:
        st.caption(md(subtitle))

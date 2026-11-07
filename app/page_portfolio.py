import plotly.graph_objects as go
import streamlit as st
from theme import ALERT, CONTEXT, HIGHLIGHT, insight, md, money, pct

from credit_risk import config
from data import report, scoring_params

st.title("A carteira por faixa de risco")
st.caption(
    "Score de 0 a 1000 (maior = menor risco; +60 pontos = chance de pagar dobra) e faixas A–E com cortes "
    "definidos no treino. Números do holdout."
)
bands = report("score_bands.csv")
ho = bands[bands["base"] == "holdout"].set_index("faixa").loc[list(config.BAND_SHARES)]
params = scoring_params()

insight(
    "A taxa de calote sobe de 6% (A) para 70% (E), sem inversões",
    "As mesmas faixas no treino e no holdout têm taxas praticamente iguais: a régua é estável.",
)
colors = [HIGHLIGHT, "#86b6ef", CONTEXT, "#eb6834", ALERT]
fig = go.Figure(
    go.Bar(
        x=[
            f"{b}<br>{int(lo)}–{int(hi)}"
            for b, lo, hi in zip(ho.index, ho["score_min"], ho["score_max"], strict=True)
        ],
        y=ho["taxa_default"],
        marker_color=colors,
        text=[pct(v, 0) for v in ho["taxa_default"]],
        textposition="outside",
    )
)
fig.update_layout(yaxis_tickformat=".0%", height=340, xaxis_title="faixa e intervalo de score")
st.plotly_chart(fig, width="stretch")

insight(
    "Tabela de política",
    "O que fazer com cada faixa. Dentro da capacidade, o valor esperado decide quem é acionado primeiro.",
)
table = {
    "faixa": list(ho.index),
    "score": [
        f"{int(lo)}–{int(hi)}" for lo, hi in zip(ho["score_min"], ho["score_max"], strict=True)
    ],
    "ação": list(ho["acao"]),
    "clientes": list(ho["clientes"]),
    "% da carteira": [pct(v) for v in ho["participacao"]],
    "taxa de calote": [pct(v) for v in ho["taxa_default"]],
    "perda esperada": [money(v) for v in ho["perda_esperada"]],
    "perda real": [money(v) for v in ho["perda_real"]],
    "prioritários": list(ho["prioritarios"]),
}
st.dataframe(table, width="stretch", hide_index=True)
st.caption(
    md(
        f"Prioritário = valor esperado de agir acima de {money(params.ev_cut, 'auto')} (equivale a acionar 10% da carteira). "
        "Há prioritários até na faixa B: risco moderado, mas dívidas grandes."
    )
)

c1, c2 = st.columns(2)
with c1:
    el = report("expected_loss_summary.csv").set_index("base").loc["holdout"]
    insight(
        f"A perda esperada acertou a perda real com erro de {pct(abs(el['el_total'] / el['perda_realizada'] - 1))}",
        "Previsão da perda total da carteira (base para provisionar).",
    )
    fig = go.Figure(
        go.Bar(
            x=["prevista (EL)", "real"],
            y=[el["el_total"] / 1e6, el["perda_realizada"] / 1e6],
            marker_color=[HIGHLIGHT, CONTEXT],
            text=[money(el["el_total"]), money(el["perda_realizada"])],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis_title="NT$ milhões", height=320)
    st.plotly_chart(fig, width="stretch")
with c2:
    curve = report("expected_loss_concentration.csv")
    curve = curve[curve["base"] == "holdout"]
    insight(
        "10% dos clientes concentram 63% da perda",
        "Ordenando pela perda esperada (azul) ou só pela probabilidade (cinza).",
    )
    fig = go.Figure()
    for crit, color in [("EL", HIGHLIGHT), ("PD", CONTEXT)]:
        part = curve[curve["ordenacao"] == crit]
        fig.add_scatter(
            x=part["fracao_clientes"],
            y=part["fracao_capturada"],
            name=f"por {crit}",
            line={"color": color, "width": 3},
        )
    fig.update_layout(
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        height=320,
        xaxis_title="clientes mais prioritários",
        yaxis_title="perda capturada",
        xaxis_range=[0, 0.5],
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    st.plotly_chart(fig, width="stretch")

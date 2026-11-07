import numpy as np
import plotly.graph_objects as go
import streamlit as st
from theme import CONTEXT, HIGHLIGHT, insight, md, money, pct

from credit_risk import config
from credit_risk.decision import (
    Premises,
    capacity_mask,
    expected_value,
    policy_result,
    rule_priority,
)
from data import artifacts

art = artifacts()
hp = art.holdout_pred.set_index("id").loc[art.holdout["id"]]
y, ead, p = hp[config.TARGET].to_numpy(), hp["ead"].to_numpy(), hp["pd"].to_numpy()

st.title("O modelo e a decisão: mexa nas premissas")
st.caption(
    "Carteira de exemplo: os 6.000 clientes do holdout (nunca vistos pelo modelo). A política oficial foi "
    "definida no treino (Etapa 12); aqui você explora o que muda se as premissas forem outras."
)

with st.sidebar:
    st.header("Premissas")
    lgd = st.slider("LGD: fração da dívida perdida no calote", 0.30, 0.90, config.LGD, 0.05)
    eff = st.slider(
        "Eficácia da ação: fração da perda evitada", 0.05, 0.80, config.ACTION_EFFICACY, 0.05
    )
    cost = st.slider("Custo da ação (NT$ por cliente)", 0, 1000, int(config.ACTION_COST), 50)
    friction = st.slider(
        "Atrito: margem perdida com bom pagador (NT$)", 0, 2000, int(config.FRICTION_COST), 50
    )
    cap = st.slider(
        "Capacidade: fração da carteira acionada por mês", 0.02, 0.40, config.CAPACITY, 0.01
    )

prem = Premises(lgd=lgd, efficacy=eff, action_cost=float(cost), friction_cost=float(friction))
ev = expected_value(p, ead, prem)
masks = {
    "Probabilidade × dívida (modelo)": capacity_mask(ev, cap, ev > 0),
    "Só a probabilidade (modelo)": capacity_mask(p, cap),
    "Regra de atraso (sem modelo)": capacity_mask(rule_priority(art.holdout), cap),
}
res = {n: policy_result(m, y, ead, prem) for n, m in masks.items()}
model_v = res["Probabilidade × dívida (modelo)"]["valor"]
rule_v = res["Regra de atraso (sem modelo)"]["valor"]
loss = res["Regra de atraso (sem modelo)"]["perda_sem_acao"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Perda se ninguém agir", money(loss))
c2.metric(
    "Perda evitada pelo modelo",
    money(model_v),
    pct(model_v / loss) + " da perda",
    delta_arrow="off",
    delta_color="off",
)
c3.metric(
    "Ganho sobre a regra de atraso",
    money(model_v - rule_v),
    pct((model_v - rule_v) / abs(rule_v)) if rule_v else None,
    delta_color="normal",
)
c4.metric(
    "Clientes acionados",
    f"{int(masks['Probabilidade × dívida (modelo)'].sum()):,}".replace(",", "."),
    help="O modelo só aciona quem tem valor esperado positivo; pode ficar abaixo da capacidade.",
)

if model_v > rule_v:
    title = f"Com estas premissas, o modelo evita {money(model_v - rule_v)} a mais que a regra de atraso"
else:
    title = "Com estas premissas, o modelo não supera a regra de atraso"
insight(
    title,
    "Mesmo número de clientes acionados para todas as políticas. Perda evitada já descontados os custos das ações.",
)
names = list(masks)
fig = go.Figure(
    go.Bar(
        x=names,
        y=[res[n]["valor"] / 1e6 for n in names],
        marker_color=[HIGHLIGHT, CONTEXT, CONTEXT],
        text=[money(res[n]["valor"]) for n in names],
        textposition="outside",
        customdata=[[res[n]["calotes_alcancados"], res[n]["acionados"]] for n in names],
        hovertemplate="%{x}<br>valor %{y:.2f} mi<br>%{customdata[0]} calotes entre %{customdata[1]} acionados<extra></extra>",
    )
)
fig.update_layout(yaxis_title="perda evitada (NT$ milhões)", height=380)
st.plotly_chart(fig, width="stretch")

insight(
    "O modelo acha menos calotes, mas os que custam mais caro",
    "Calotes alcançados por cada política. Escolher pelo valor em jogo, e não só pela chance, é o que gera o dinheiro.",
)
st.dataframe(
    {
        "política": names,
        "acionados": [res[n]["acionados"] for n in names],
        "calotes alcançados": [res[n]["calotes_alcancados"] for n in names],
        "% dos calotes": [pct(res[n]["pct_calotes_alcancados"]) for n in names],
        "perda evitada": [money(res[n]["valor"]) for n in names],
    },
    width="stretch",
    hide_index=True,
)
bill = 10_000
be = (
    (cost + friction) / (lgd * bill * eff + friction) if lgd * bill * eff + friction > 0 else np.nan
)
st.caption(
    md(
        f"Com estas premissas, para uma fatura de NT$ 10 mil, agir compensa a partir de {pct(min(be, 1), 0)} de chance de calote."
    )
)

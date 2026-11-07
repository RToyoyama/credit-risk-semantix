import plotly.graph_objects as go
import streamlit as st
from theme import ALERT, CONTEXT, insight

from credit_risk import config
from data import report_json

st.title("Monitoramento: o modelo ainda vale para os clientes de hoje?")
st.caption(
    "Todo mês, cada lote novo é comparado com a população de treino. Aqui: o holdout e três cenários simulados. "
    "PSI < 0,10 estável · 0,10–0,25 atenção · > 0,25 investigar."
)
reports = report_json("monitoring.json")
lote = st.selectbox("Lote", [r["lote"] for r in reports])
r = next(x for x in reports if x["lote"] == lote)
st.metric(
    "Status geral",
    r["status_geral"].capitalize(),
    f"{r['clientes']} clientes",
    delta_arrow="off",
    delta_color="off",
)

c1, c2 = st.columns(2)
for col, (key, label, limit) in zip(
    [c1, c2],
    [
        ("pd_media", "PD média ÷ referência", config.PD_MEAN_RATIO_MAX),
        ("fracao_faixa_e", "Faixa E ÷ referência", config.BAND_E_RATIO_MAX),
    ],
    strict=True,
):
    ind = r["indicadores"][key]
    col.metric(
        label,
        f"{ind['razao']:.2f}×".replace(".", ","),
        f"alerta acima de {limit:.1f}×".replace(".", ","),
        delta_arrow="off",
        delta_color="off",
    )

insight(
    "PSI por variável",
    "O PSI pode não ver mudanças concentradas nos clientes mais arriscados (como no cenário de crise): "
    "por isso os dois indicadores acima completam o monitor.",
)
var = r["variaveis"]
names = list(var)
vals = [var[n]["psi"] for n in names]
colors = [ALERT if v > 0.25 else ("#eda100" if v > 0.10 else CONTEXT) for v in vals]
fig = go.Figure(
    go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}".replace(".", ",") for v in vals],
        textposition="outside",
    )
)
fig.add_vline(x=0.10, line_dash="dot", annotation_text="atenção")
fig.add_vline(x=0.25, line_dash="dot", annotation_text="investigar")
fig.update_layout(
    height=380,
    xaxis_title="PSI",
    yaxis={"autorange": "reversed"},
    xaxis_range=[0, max(0.35, max(vals) * 1.15)],
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Cores: vermelho = investigar, amarelo = atenção, cinza = estável. Rotina sugerida: atenção → revisar as "
    "variáveis que mudaram; investigar → congelar mudanças de política, comparar a PD com o calote real quando "
    "ele chegar e decidir entre recalibrar e retreinar."
)

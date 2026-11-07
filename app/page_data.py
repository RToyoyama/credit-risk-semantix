import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from theme import CONTEXT, HIGHLIGHT, insight, pct

from credit_risk import config
from credit_risk.stats import rate_by_group
from data import artifacts, report

train = artifacts().train
y = train[config.TARGET]
st.title("O que os dados revelam")
st.caption(
    "Análise feita só no treino (24.000 clientes). Detalhes e testes estatísticos no notebook 01."
)

insight(
    "Com 2 meses de atraso em setembro, 69% dão calote em outubro",
    "Taxa de calote por situação do pagamento em setembro (≤ 0 = em dia; 1+ = meses de atraso). "
    "É o sinal mais forte de toda a base.",
)
rates = rate_by_group(train.assign(s=train["pay_status_1"].clip(upper=4)), "s")
labels = {-2: "-2", -1: "-1", 0: "0", 1: "1", 2: "2", 3: "3", 4: "4+"}
fig = go.Figure(
    go.Bar(
        x=[labels[s] for s in rates["s"]],
        y=rates["taxa"],
        marker_color=[HIGHLIGHT if s >= 2 else CONTEXT for s in rates["s"]],
        error_y={
            "type": "data",
            "symmetric": False,
            "array": rates["ic_sup"] - rates["taxa"],
            "arrayminus": rates["taxa"] - rates["ic_inf"],
            "color": "#52514e",
        },
        customdata=rates["n"],
        hovertemplate="status %{x}<br>calote %{y:.1%}<br>%{customdata} clientes<extra></extra>",
    )
)
fig.add_hline(
    y=y.mean(),
    line_dash="dash",
    line_color="#52514e",
    annotation_text=f"média {pct(y.mean())}",
    annotation_position="top left",
)
for i, (top, v) in enumerate(zip(rates["ic_sup"], rates["taxa"], strict=True)):
    fig.add_annotation(x=i, y=top, text=pct(v, 0), showarrow=False, yshift=12)
fig.update_layout(
    yaxis_tickformat=".0%",
    xaxis_title="status de pagamento em setembro",
    height=380,
    xaxis_type="category",
)
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
with c1:
    insight(
        "Limite baixo, mais calote: 32% contra 13%",
        "Taxa de calote por quinto de limite de crédito.",
    )
    q = pd.qcut(train["limit_bal"], 5)
    lim = rate_by_group(train.assign(q=q), "q")
    fig = go.Figure(
        go.Bar(
            x=[f"{iv.left / 1e3:.0f}–{iv.right / 1e3:.0f} mil" for iv in lim["q"]],
            y=lim["taxa"],
            marker_color=[HIGHLIGHT] + [CONTEXT] * 4,
            text=[pct(v, 0) for v in lim["taxa"]],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title="limite (NT$)", height=320)
    st.plotly_chart(fig, width="stretch")
with c2:
    insight(
        "Acurácia engana: todos os modelos parecem iguais",
        '"Ninguém dá calote" acerta 78% sem saber nada. O PR AUC separa os modelos.',
    )
    duel = report("model_duel_summary.csv")
    duel = duel[duel["modelo"].isin(["xgboost", "logistic", "dummy"])]
    fig = go.Figure()
    fig.add_bar(name="acurácia", x=duel["nome"], y=duel["accuracy_media"], marker_color=CONTEXT)
    fig.add_bar(name="PR AUC", x=duel["nome"], y=duel["pr_auc_media"], marker_color=HIGHLIGHT)
    fig.update_layout(
        barmode="group", yaxis_tickformat=".0%", height=320, legend={"orientation": "h"}
    )
    st.plotly_chart(fig, width="stretch")

insight(
    "Sexo, idade, escolaridade e estado civil quase não mudam o risco",
    "Tamanho de efeito de cada variável (abaixo de 0,10 = desprezível). Com 24 mil clientes, "
    'quase tudo é "estatisticamente significativo"; o efeito diz o que importa.',
)
eff = report("eda_tests.csv")
eff = eff.assign(abs_effect=eff["effect"].abs()).sort_values("abs_effect")
fig = go.Figure(
    go.Bar(
        x=eff["abs_effect"],
        y=eff["variable"],
        orientation="h",
        marker_color=[HIGHLIGHT if r else CONTEXT for r in eff["relevante"]],
    )
)
fig.add_vline(x=0.10, line_dash="dot", annotation_text="efeito pequeno")
fig.update_layout(height=760, xaxis_title="tamanho de efeito", yaxis_title=None)
st.plotly_chart(fig, width="stretch")
st.info(
    "Por isso o modelo final **não usa sexo, estado civil nem idade**: tirá-los não mudou o desempenho "
    "(Etapa 13), e a API nem aceita esses dados."
)

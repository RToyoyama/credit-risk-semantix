import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from theme import CONTEXT, HIGHLIGHT, insight, md, money, pct

from credit_risk import config
from credit_risk.decision import Premises, expected_value
from data import artifacts, report

art = artifacts()
full = pd.concat([art.train, art.holdout])
ead = full["bill_amt_1"].clip(lower=0)
loss = (config.LGD * ead * full[config.TARGET]).sum()
policy = report("holdout_policy.csv").set_index("política")
boot = report("holdout_policy_bootstrap.csv").set_index("comparação")

st.title("Quem vai deixar de pagar, quanto isso custa e o que fazer")
st.markdown(
    "Uma emissora de cartão precisa decidir, **todo mês**, em quais clientes agir antes do calote "
    "(renegociar, reduzir limite, contato preventivo). Este painel mostra como um modelo transforma "
    "essa decisão em dinheiro."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes na base", f"{len(full):,}".replace(",", "."), help="Taiwan, abr–set/2005 (UCI)")
c2.metric("Deram calote no mês seguinte", pct(full[config.TARGET].mean()))
c3.metric(
    "Perda se ninguém agir",
    money(loss),
    help=f"LGD {pct(config.LGD, 0)} × fatura de quem deu calote",
)
gain = boot.loc["valor esperado − regra de atraso", "diferença"]
c4.metric(
    "Ganho do modelo sobre a regra de atraso",
    money(gain),
    help="No holdout (6.000 clientes nunca vistos), acionando 10% da carteira",
)

insight(
    f"Com o mesmo esforço, o modelo evita {pct(gain / policy.loc['regra de atraso (sem modelo)', 'valor'], 0)} "
    'a mais de perda que a regra "mais atrasados primeiro"',
    "Prova final: 6.000 clientes que o modelo nunca viu. Intervalo de confiança de 95% da diferença: "
    f"{money(boot.loc['valor esperado − regra de atraso', 'ic_inf'])} a "
    f"{money(boot.loc['valor esperado − regra de atraso', 'ic_sup'])}.",
)
names = {
    "regra de atraso (sem modelo)": "Regra de atraso<br>(sem modelo)",
    "maior PD (modelo)": "Só a probabilidade<br>(modelo)",
    "maior valor esperado (modelo)": "Probabilidade × dívida<br>(modelo, política adotada)",
}
fig = go.Figure(
    go.Bar(
        x=[names[n] for n in names],
        y=[policy.loc[n, "valor"] / 1e6 for n in names],
        marker_color=[CONTEXT, CONTEXT, HIGHLIGHT],
        text=[money(policy.loc[n, "valor"]) for n in names],
        textposition="outside",
    )
)
fig.update_layout(yaxis_title="perda evitada (NT$ milhões)", height=360, showlegend=False)
st.plotly_chart(fig, width="stretch")

insight(
    "Os dois erros não custam igual",
    "Deixar passar um caloteiro custa caro; incomodar um bom pagador custa pouco. Teste com os seus números:",
)
col_a, col_b = st.columns([1, 2])
with col_a:
    bill = st.number_input("Fatura do cliente (NT$)", 0, 500_000, 10_000, step=1_000)
    p = st.slider("Chance de calote (PD)", 0.0, 1.0, 0.20, 0.01, format="%.2f")
prem = Premises()
value = float(expected_value([p], [bill], prem)[0])
saved = config.LGD * bill * config.ACTION_EFFICACY - config.ACTION_COST
cost = config.ACTION_COST + config.FRICTION_COST
avoided = config.LGD * bill * config.ACTION_EFFICACY
breakeven = cost / (avoided + config.FRICTION_COST) if avoided + config.FRICTION_COST > 0 else 1.0
with col_b:
    st.markdown(
        md(
            f"- Se ele **daria calote** e a gente age: evitamos **{money(saved, 'auto')}** "
            f"({pct(config.LGD, 0)} de perda × {pct(config.ACTION_EFFICACY, 0)} evitados − custo da ação).\n"
            f"- Se ele **pagaria** e a gente age: perdemos **{money(cost, 'auto')}** (ação + atrito).\n"
            f"- **Valor esperado de agir:** {money(value, 'auto')} → "
            f"{'**vale agir**' if value > 0 else '**não vale agir**'}.\n"
            f"- Para essa fatura, agir compensa a partir de **{pct(min(breakeven, 1), 0)}** de chance de calote."
        )
    )
st.caption(
    md(
        "Premissas (configuráveis em `config.py` e testadas na análise de sensibilidade): "
        f"LGD {pct(config.LGD, 0)}, eficácia da ação {pct(config.ACTION_EFFICACY, 0)}, "
        f"custo da ação NT$ {config.ACTION_COST:.0f}, atrito NT$ {config.FRICTION_COST:.0f}."
    )
)

import pandas as pd
import streamlit as st
from theme import insight, md, money, pct

from credit_risk import config
from credit_risk.reasons import ReasonExplainer
from credit_risk.scoring import score_clients
from data import artifacts, scoring_params

PRESETS = {
    "Bom pagador": {
        "limit_bal": 200_000,
        "education": 1,
        "status": [-1] * 6,
        "bill": [3_000] * 6,
        "paid": [3_000] * 6,
    },
    "Rotativo em dia": {
        "limit_bal": 100_000,
        "education": 2,
        "status": [0] * 6,
        "bill": [70_000, 68_000, 66_000, 60_000, 55_000, 50_000],
        "paid": [3_000] * 6,
    },
    "Atraso recente": {
        "limit_bal": 50_000,
        "education": 2,
        "status": [2, 2, 0, 0, 0, 0],
        "bill": [48_500, 47_000, 46_200, 30_000, 28_000, 25_000],
        "paid": [0, 1_500, 2_000, 1_500, 1_000, 1_000],
    },
}
MONTHS = ["set", "ago", "jul", "jun", "mai", "abr"]


@st.cache_resource
def explainer() -> ReasonExplainer:
    return ReasonExplainer(artifacts().model)


st.title("Simulador de cliente")
st.caption(
    "Mesma regra da API: PD → perda esperada → score → faixa → ação → motivos. "
    "Sexo, estado civil e idade não são pedidos: o modelo não os usa."
)
preset = PRESETS[st.radio("Começar de um perfil", list(PRESETS), horizontal=True)]

with st.form("cliente"):
    c1, c2 = st.columns(2)
    limit = c1.number_input(
        "Limite de crédito (NT$)", 10_000, 1_000_000, preset["limit_bal"], step=10_000
    )
    edu = c2.selectbox(
        "Escolaridade",
        [1, 2, 3, 4],
        index=preset["education"] - 1,
        format_func=lambda v: config.CATEGORY_LABELS["education"][v],
    )
    st.markdown("**Situação do pagamento** (≤ 0 = em dia; 1+ = meses de atraso)")
    status = [
        c.number_input(m, -2, 9, preset["status"][i], key=f"s{i}")
        for i, (c, m) in enumerate(zip(st.columns(6), MONTHS, strict=True))
    ]
    st.markdown(md("**Fatura do mês (NT$)**"))
    bill = [
        c.number_input(m, -200_000, 1_000_000, preset["bill"][i], step=1_000, key=f"b{i}")
        for i, (c, m) in enumerate(zip(st.columns(6), MONTHS, strict=True))
    ]
    st.markdown(md("**Valor pago no mês (NT$)**"))
    paid = [
        c.number_input(m, 0, 1_000_000, preset["paid"][i], step=500, key=f"p{i}")
        for i, (c, m) in enumerate(zip(st.columns(6), MONTHS, strict=True))
    ]
    submitted = st.form_submit_button("Calcular", type="primary")

client = pd.DataFrame(
    [
        {
            "limit_bal": limit,
            "education": edu,
            **{f"pay_status_{i + 1}": s for i, s in enumerate(status)},
            **{f"bill_amt_{i + 1}": b for i, b in enumerate(bill)},
            **{f"pay_amt_{i + 1}": p for i, p in enumerate(paid)},
        }
    ]
)
out = score_clients(artifacts().model, client, scoring_params(), explainer()).iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Score", f"{out['score']}", help="0 a 1000; +60 pontos = chance de pagar dobra")
c2.metric("Faixa", out["faixa"])
c3.metric("Chance de calote", pct(out["pd"]))
c4.metric("Perda esperada", money(out["perda_esperada"], "auto"))
insight(f"Ação: {out['acao']}" + (" · prioritário para acionar" if out["prioritario"] else ""))
st.markdown("**Por que essa nota?**")
for reason in out["motivos"]:
    st.markdown(md(f"- {reason}"))

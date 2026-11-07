import plotly.graph_objects as go
import streamlit as st
from theme import CATEGORICAL, insight, pct

from credit_risk import config
from credit_risk.features import behavior_features
from credit_risk.stats import rate_by_group
from data import artifacts, report, train_segments

train = artifacts().train
seg = train_segments()
names = config.SEGMENT_NAMES
st.title("Três jeitos de usar o cartão")
st.caption(
    "Grupos formados por comportamento (PCA + K-means), sem saber quem deu calote e sem dados pessoais. "
    "Detalhes no notebook 02."
)

rates = rate_by_group(train.assign(segment=seg), "segment")
profile = report("segment_profiles.csv").set_index("segment")
cols = st.columns(3)
for col, (s, row) in zip(cols, rates.set_index("segment").iterrows(), strict=True):
    col.metric(
        names[s].capitalize(),
        pct(row["taxa"]),
        f"{pct(profile.loc[s, 'participacao'], 0)} dos clientes",
        delta_color="off",
        delta_arrow="off",
        help="taxa de calote no mês seguinte",
    )

insight(
    "O risco vem do atraso, não do uso do rotativo",
    "Quem vive no rotativo (usa ~70% do limite e paga pouco) tem calote parecido com o de quem quita tudo, "
    "enquanto não atrasa. Cortar o rotativo cortaria receita sem reduzir perda.",
)
b = behavior_features(train)
months = ["abr", "mai", "jun", "jul", "ago", "set"]
tabs = st.tabs(["Uso do limite", "Fração paga da fatura", "Meses de atraso"])
series = [config.UTILIZATION, config.PAYMENT_RATIO, config.DELAY]
for tab, cols_ in zip(tabs, series, strict=True):
    with tab:
        means = b[cols_].groupby(seg.to_numpy()).mean()
        fig = go.Figure()
        for s, color in zip(means.index, CATEGORICAL, strict=True):
            fig.add_scatter(
                x=months[-len(cols_) :],
                y=means.loc[s].to_numpy()[::-1],
                name=names[s],
                line={"color": color, "width": 3},
                mode="lines+markers",
            )
        fig.update_layout(
            height=340,
            legend={"orientation": "h"},
            yaxis_tickformat=".0%" if cols_ != config.DELAY else None,
        )
        st.plotly_chart(fig, width="stretch")

with st.expander("Perfil completo dos segmentos"):
    show = profile.copy()
    show.index = show["nome"]
    st.dataframe(show.drop(columns="nome").style.format(precision=2), width="stretch")
st.info(
    "Os segmentos **não entram no modelo** (não melhoraram a previsão), mas ajudam a explicar a carteira."
)

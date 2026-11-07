"""Dashboard de risco de crédito (Streamlit).

Uso: ``make app`` e abrir http://localhost:8501.
"""

import streamlit as st

st.set_page_config(page_title="Risco de crédito", page_icon="💳", layout="wide")

pages = [
    st.Page("page_problem.py", title="1. O problema", default=True),
    st.Page("page_data.py", title="2. O que os dados revelam"),
    st.Page("page_segments.py", title="3. Os segmentos"),
]
st.navigation(pages).run()

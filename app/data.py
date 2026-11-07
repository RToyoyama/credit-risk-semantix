"""Acesso a dados do dashboard: artefatos (com cache) e resultados salvos em reports/."""

import json

import pandas as pd
import streamlit as st

from credit_risk import config
from credit_risk.artifacts import Artifacts, ensure_artifacts
from credit_risk.scoring import ScoringParams, load_params
from credit_risk.segmentation import SegmentEncoder


@st.cache_resource(
    show_spinner="Preparando dados e modelo (na primeira visita, menos de 1 minuto)..."
)
def artifacts() -> Artifacts:
    return ensure_artifacts()


@st.cache_resource
def scoring_params() -> ScoringParams:
    return load_params()


@st.cache_data
def report(name: str) -> pd.DataFrame:
    return pd.read_csv(config.REPORTS_DIR / name)


@st.cache_data
def report_json(name: str) -> dict | list:
    return json.loads((config.REPORTS_DIR / name).read_text())


@st.cache_data
def train_segments() -> pd.Series:
    train = artifacts().train
    return pd.Series(SegmentEncoder().fit(train).predict(train), index=train.index, name="segment")

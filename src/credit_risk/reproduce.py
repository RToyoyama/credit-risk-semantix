"""Confere se um ambiente novo reproduz as métricas do holdout salvas no Git.

Reconstrói dados, modelo e previsões (``artifacts.ensure_artifacts``) e
compara as métricas do holdout com ``reports/holdout_metrics.csv``. Usado
para validar o container Docker.

Uso: ``python -m credit_risk.reproduce`` (ou ``make docker-reproduce``).
"""

import sys

import pandas as pd

from credit_risk import config
from credit_risk.artifacts import ensure_artifacts
from credit_risk.final import METRICS

TOLERANCE = 1e-6


def compare() -> pd.DataFrame:
    """Métricas recalculadas × salvas.

    Returns:
        Uma linha por métrica: salvo, recalculado e diferença.
    """
    art = ensure_artifacts()
    hp = art.holdout_pred
    saved = pd.read_csv(config.HOLDOUT_METRICS).set_index("métrica")["valor"]
    rows = [
        {"métrica": name, "salvo": saved[name], "recalculado": fn(hp[config.TARGET], hp["pd"])}
        for name, fn in METRICS.items()
    ]
    out = pd.DataFrame(rows)
    out["diferenca"] = (out["recalculado"] - out["salvo"]).abs()
    return out


if __name__ == "__main__":
    table = compare()
    print(table.to_string(index=False))
    ok = bool((table["diferenca"] < TOLERANCE).all())
    print("REPRODUZIDO" if ok else "DIVERGENTE")
    sys.exit(0 if ok else 1)

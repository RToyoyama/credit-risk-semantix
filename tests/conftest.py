import numpy as np
import pandas as pd
import pytest

from credit_risk import config


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Tabela sintética pequena, válida segundo o schema da UCI."""
    data = {
        "ID": [1, 2, 3, 4],
        "LIMIT_BAL": [20_000, 50_000, 120_000, 500_000],
        "SEX": [1, 2, 2, 1],
        "EDUCATION": [1, 2, 3, 4],
        "MARRIAGE": [1, 2, 3, 1],
        "AGE": [21, 35, 47, 79],
        **{c: [-2, -1, 0, 2] for c in config.PAY_STATUS_COLS},
        **{c: [-500, 0, 3_000, 90_000] for c in config.BILL_AMT_COLS},
        **{c: [0, 100, 2_000, 5_000] for c in config.PAY_AMT_COLS},
        config.RAW_TARGET: [0, 0, 1, 1],
    }
    return pd.DataFrame(data)[config.RAW_COLUMNS]


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """Tabela limpa sintética com 1.000 clientes e ~22% de default."""
    rng = np.random.default_rng(0)
    n = 1_000
    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "age": rng.integers(21, 80, n),
            config.TARGET: (rng.random(n) < 0.22).astype(int),
        }
    )

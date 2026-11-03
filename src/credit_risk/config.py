"""Configuração central: caminhos, fonte dos dados, seeds e schema esperado."""

from pathlib import Path

# ── Caminhos ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── Fonte dos dados (UCI, dataset 350, CC BY 4.0) ───────────
RAW_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
RAW_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
RAW_ZIP = RAW_DIR / "default_of_credit_card_clients.zip"
RAW_XLS_NAME = "default of credit card clients.xls"
RAW_PARQUET = RAW_DIR / "credit_default_raw.parquet"
DOWNLOAD_TIMEOUT_S = 60

# ── Reprodutibilidade ───────────────────────────────────────
SEED = 42

# ── Schema da camada raw (nomes originais da UCI) ───────────
ID_COL = "ID"
RAW_TARGET = "default payment next month"
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_COLUMNS = [
    ID_COL,
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS_COLS,
    *BILL_AMT_COLS,
    *PAY_AMT_COLS,
    RAW_TARGET,
]

# Valores que a documentação da UCI descreve.
DOCUMENTED_CATEGORIES: dict[str, set[int]] = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
    RAW_TARGET: {0, 1},
}
# Valores não documentados, mas sabidamente presentes: geram aviso, não erro
# (tratados em cleaning.py, Etapa 4).
KNOWN_UNDOCUMENTED: dict[str, set[int]] = {
    "EDUCATION": {0, 5, 6},
    "MARRIAGE": {0},
}
# Faixas numéricas válidas (inclusivas). None = sem limite.
VALID_RANGES: dict[str, tuple[float | None, float | None]] = {
    "LIMIT_BAL": (1, None),
    "AGE": (18, 100),
    **dict.fromkeys(PAY_STATUS_COLS, (-2, 9)),
    **dict.fromkeys(PAY_AMT_COLS, (0, None)),
}

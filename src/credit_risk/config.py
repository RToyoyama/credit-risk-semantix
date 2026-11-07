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

# ── Camadas interim/processed ───────────────────────────────
INTERIM_PARQUET = INTERIM_DIR / "credit_default_clean.parquet"
TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
HOLDOUT_PARQUET = PROCESSED_DIR / "holdout.parquet"
HOLDOUT_IDS = REPORTS_DIR / "holdout_ids.csv"  # versionado: define o holdout para sempre
HOLDOUT_SIZE = 0.20

# ── Schema limpo (snake_case; índice 1 = set/2005, o mês mais recente) ──
N_MONTHS = 6
TARGET = "default"
PAY_STATUS = [f"pay_status_{i}" for i in range(1, N_MONTHS + 1)]
BILL_AMT = [f"bill_amt_{i}" for i in range(1, N_MONTHS + 1)]
PAY_AMT = [f"pay_amt_{i}" for i in range(1, N_MONTHS + 1)]
RENAME_MAP: dict[str, str] = {
    ID_COL: "id",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    **dict(zip(PAY_STATUS_COLS, PAY_STATUS, strict=True)),
    **dict(zip(BILL_AMT_COLS, BILL_AMT, strict=True)),
    **dict(zip(PAY_AMT_COLS, PAY_AMT, strict=True)),
    RAW_TARGET: TARGET,
}
# Valores não documentados → categoria "outros" já existente.
EDUCATION_OTHERS = 4
MARRIAGE_OTHERS = 3
CATEGORY_LABELS: dict[str, dict[int, str]] = {
    "sex": {1: "masculino", 2: "feminino"},
    "education": {1: "pós-graduação", 2: "universidade", 3: "ensino médio", 4: "outros"},
    "marriage": {1: "casado", 2: "solteiro", 3: "outros"},
}

# ── Inferência estatística (critérios fixados antes da EDA) ──
ALPHA = 0.05
N_BOOT = 10_000
CI_LEVEL = 0.95
# Cohen: abaixo de 0,10 desprezível; 0,10 pequeno; 0,30 médio; 0,50 grande.
EFFECT_THRESHOLDS: dict[str, float] = {"pequeno": 0.10, "médio": 0.30, "grande": 0.50}

# ── Features comportamentais (regras fixas, sem alvo) ───────
UTILIZATION_CLIP: tuple[float, float] = (0.0, 1.5)  # 99% dos clientes usam <= 109% do limite
PAYMENT_RATIO_CLIP: tuple[float, float] = (0.0, 1.0)  # pagar mais que a fatura = quitou
UTILIZATION = [f"utilization_{i}" for i in range(1, N_MONTHS + 1)]
PAYMENT_RATIO = [f"payment_ratio_{i}" for i in range(1, N_MONTHS)]  # mês 6 não tem fatura anterior
DELAY = [f"delay_{i}" for i in range(1, N_MONTHS + 1)]
BEHAVIOR_FEATURES = UTILIZATION + PAYMENT_RATIO + DELAY

# ── Segmentação ─────────────────────────────────────────────
PCA_VARIANCE_TARGET = 0.80  # menor nº de componentes que explique >= 80%
PCA_N_COMPONENTS = 4  # resultado do critério acima (82,3%, Etapa 6)
K_RANGE = range(3, 9)
MIN_SEGMENT_SHARE = 0.05
ARI_MIN = 0.80
SILHOUETTE_SAMPLE = 8_000  # silhueta é O(n²): calculada numa amostra fixa
N_SEGMENTS = 3  # melhor silhueta com todo grupo >= 5% (Etapa 6)
SEGMENT_NAMES: dict[int, str] = {0: "quitadores", 1: "rotativo em dia", 2: "atraso crônico"}
KMEANS_N_INIT = 10

# ── Validação cruzada ───────────────────────────────────────
CV_SPLITS = 5
CV_REPEATS = 3
WIN_RATE_MIN = 0.60  # para adotar uma mudança: PR AUC sobe com win rate > 60%

# ── Features de resumo (Etapa 7) ────────────────────────────
RECENT_MONTHS = 3  # set–jul = recente; jun–abr = antigo
CATEGORICAL_COLS = ["sex", "education", "marriage"]
RAW_NUMERIC_COLS = ["limit_bal", "age", *PAY_STATUS, *BILL_AMT, *PAY_AMT]
ENGINEERED_FEATURES = [
    "log_limit",
    "util_mean",
    "util_recent",
    "util_trend",
    "util_volatility",
    "pay_ratio_mean",
    "pay_ratio_recent",
    "paid_full_all",
    "delay_max",
    "delay_months",
    "delay_recent",
    "delay_trend",
    "revolving_months",
]
MIN_CATEGORY_COUNT = 50  # categorias com menos casos no treino viram "infrequente"
# Etapa 8: pay_status como categoria nos modelos lineares (PR AUC +0,016, win rate 100%).
# Árvores mantêm o número: cortam em qualquer ponto, então a escala não as engana.
STATUS_AS_CATEGORY_LINEAR = True
# Etapa 9: logística enxuta para interpretar razões de chances (todas com VIF < 3).
INTERPRETABLE_FEATURES = [
    "delay_recent",
    "delay_months",
    "log_limit",
    "util_mean",
    "pay_ratio_mean",
]
VIF_ALERT = 5.0
BALANCING_STRATEGY = "none"  # Etapa 10: nenhuma técnica passou (Brier piora em 100% dos folds)

# ── Tuning (Etapa 11) ───────────────────────────────────────
SEARCH_SEED = 2024  # folds da busca ≠ folds da avaliação (seed 42)
SEARCH_N_ITER = {"random_forest": 40, "xgboost": 60}
BEST_PARAMS = REPORTS_DIR / "best_params.json"
FINAL_MODEL = "xgboost"  # Etapa 11: líder após o tuning; nenhum ensemble passou no critério
CALIBRATION_METHOD = (
    "none"  # Etapa 12A: sigmoid piora o Brier; isotônica piora o PR AUC (win rate 40%)
)

# ── Premissas de negócio (Etapa 12B; aceitas em 2026-11-07, com sensibilidade) ──
LGD = 0.60  # fração da dívida perdida no calote (cartão sem garantia: 45%–75%)
ACTION_EFFICACY = 0.30  # fração da perda que a ação preventiva evita num caloteiro
ACTION_COST = 100.0  # NT$ por cliente acionado (contato, análise)
FRICTION_COST = 300.0  # NT$ de margem perdida ao acionar um bom pagador
SENSITIVITY: dict[str, list[float]] = {
    "lgd": [0.45, 0.60, 0.75],
    "efficacy": [0.15, 0.30, 0.60],
    "action_cost": [50.0, 100.0, 200.0],
    "friction_cost": [150.0, 300.0, 600.0],
}
OOF_PREDICTIONS = PROCESSED_DIR / "oof_predictions.parquet"
# Restrição operacional (Etapa 12B, decidida após ver que a política sem limite
# aciona ~60% da carteira): a equipe aciona no máximo esta fração por mês.
CAPACITY = 0.10
CAPACITY_SENSITIVITY = [0.05, 0.10, 0.20]

# ── Atributos sensíveis (Etapa 13A) ─────────────────────────
SENSITIVE_VARIANTS: dict[str, list[str]] = {
    "atual": [],
    "sem sexo e estado civil": ["sex", "marriage"],
    "sem sexo, estado civil e idade": ["sex", "marriage", "age"],
}
NONINFERIORITY_PR_AUC = 0.005  # perda máxima aceitável de PR AUC ao remover atributos
NONINFERIORITY_VALUE = 0.02  # perda máxima aceitável (fração) do valor da política
MODEL_EXCLUDE: list[str] = [
    "sex",
    "marriage",
    "age",
]  # Etapa 13A: removidos sem perda (não inferioridade)

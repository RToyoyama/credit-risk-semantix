# Imagem única para a API (FastAPI) e o dashboard (Streamlit).
# Dados e modelo não entram na imagem: são reconstruídos a partir da UCI na
# primeira execução (credit_risk.artifacts) e guardados num volume.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1 \
    CREDIT_RISK_ROOT=/app

# libgomp1: paralelismo do XGBoost; curl: healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependências primeiro (camada reaproveitada enquanto o lock não mudar).
COPY requirements.lock ./
RUN uv pip install -r requirements.lock

COPY pyproject.toml README.md LICENSE Makefile ./
COPY src ./src
RUN uv pip install -e . --no-deps

COPY api ./api
COPY app ./app
COPY reports ./reports
COPY tests ./tests
COPY .streamlit ./.streamlit

# Usuário sem privilégios; pastas de dados e modelo graváveis por ele.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p data/raw data/interim data/processed models \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

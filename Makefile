.DEFAULT_GOAL := help
PY := .venv/bin/python
BIN := .venv/bin

.PHONY: help venv install lock data split experiments tune final monitor api app docker-build docker-test docker-reproduce up down lint format test check hooks clean

help: ## Lista os comandos disponíveis
	@grep -E '^[a-z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

venv: ## Cria o ambiente virtual com Python 3.11
	uv venv --python 3.11 --seed .venv

install: ## Instala as dependências exatas do lock + pacote editável + hooks
	uv pip sync --python $(PY) requirements.lock
	uv pip install --python $(PY) -e . --no-deps
	$(BIN)/pre-commit install

lock: ## Regenera o requirements.lock a partir do pyproject.toml
	uv pip compile pyproject.toml --extra dev --python-version 3.11 --python-platform x86_64-manylinux_2_28 -o requirements.lock

data: ## Baixa os dados da UCI, confere o SHA-256, valida e salva em data/raw/
	$(PY) -m credit_risk.data

split: ## Limpa os dados (interim) e separa treino/holdout (processed)
	$(PY) -m credit_risk.cleaning
	$(PY) -m credit_risk.split

experiments: ## Roda os experimentos pareados e salva em reports/
	$(PY) -m credit_risk.experiments

tune: ## Ajusta hiperparâmetros (seed de busca separada) e salva best_params.json
	$(PY) -m credit_risk.tuning

final: ## Treina o modelo final e avalia o holdout UMA vez (recusa rodar de novo)
	$(PY) -m credit_risk.final

monitor: ## PSI do holdout e de cenários de drift; gera reports/monitoring.json
	$(PY) -m credit_risk.monitoring

api: ## Sobe a API de scoring em http://localhost:8000 (docs em /docs)
	$(BIN)/uvicorn api.main:app --host 0.0.0.0 --port 8000

app: ## Sobe o dashboard em http://localhost:8501
	$(BIN)/streamlit run app/dashboard.py --server.port 8501

docker-build: ## Constrói a imagem Docker
	docker compose build

docker-test: ## Roda lint e testes dentro do container
	docker compose run --rm --no-deps api sh -c "ruff check . && pytest"

docker-reproduce: ## Reconstrói tudo no container e confere as métricas do holdout salvas
	docker compose run --rm --no-deps api python -m credit_risk.reproduce

up: ## Sobe API (8000) e dashboard (8501) com Docker
	docker compose up -d

down: ## Derruba os containers (mantém os volumes de dados e modelo)
	docker compose down

lint: ## Verifica o código com ruff (sem alterar)
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## Corrige e formata o código com ruff
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test: ## Roda os testes
	$(PY) -m pytest

hooks: ## Roda todos os hooks do pre-commit
	$(BIN)/pre-commit run --all-files

check: lint test ## Lint + testes (o mesmo que a CI vai rodar)

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +

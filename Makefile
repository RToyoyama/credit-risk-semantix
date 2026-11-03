.DEFAULT_GOAL := help
PY := .venv/bin/python
BIN := .venv/bin

.PHONY: help venv install lock data split lint format test check hooks clean

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

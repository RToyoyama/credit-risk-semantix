"""API de scoring de risco de crédito (FastAPI).

Toda regra de negócio vem de ``credit_risk.scoring`` (fonte única): a API só
converte JSON em tabela, chama ``score_clients`` e devolve o resultado.

Uso: ``make api`` e abrir http://localhost:8000/docs.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request

from api.schemas import BatchIn, BatchOut, ClientIn, ModelInfo, ScoreOut
from credit_risk import __version__, config
from credit_risk.artifacts import ensure_artifacts
from credit_risk.reasons import ReasonExplainer
from credit_risk.scoring import ScoringParams, load_params, score_clients

API_VERSION = "1.0"


def _load_artifacts() -> dict[str, Any]:
    """Carrega o modelo (reconstruindo dados e modelo se faltarem), os cortes e os metadados."""
    model = ensure_artifacts().model
    meta = json.loads(config.MODEL_METADATA.read_text()) if config.MODEL_METADATA.exists() else {}
    meta.setdefault("versao", __version__)
    return {"model": model, "params": load_params(), "metadata": meta}


def _to_frame(clients: list[ClientIn]) -> pd.DataFrame:
    rows = [c.model_dump() for c in clients]
    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype("Int64")
    return df


def create_app(
    model: object | None = None,
    params: ScoringParams | None = None,
    metadata: dict[str, Any] | None = None,
) -> FastAPI:
    """Monta a API. Sem argumentos, carrega os artefatos salvos ao iniciar.

    Args:
        model: Pipeline final ajustado (para testes).
        params: Cortes do scoring (para testes).
        metadata: Metadados do modelo (para testes).

    Returns:
        Aplicação FastAPI.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        art = (
            {"model": model, "params": params, "metadata": metadata or {}}
            if model is not None
            else _load_artifacts()
        )
        art["explainer"] = ReasonExplainer(art["model"])
        app.state.art = art
        yield

    app = FastAPI(
        title="API de risco de crédito",
        version=API_VERSION,
        description=(
            "Probabilidade de calote, perda esperada, score 0-1000, faixa A-E, ação e "
            "motivos da decisão para clientes de cartão de crédito. "
            "Não coleta sexo, estado civil nem idade."
        ),
        lifespan=lifespan,
    )

    def _score(request: Request, clients: list[ClientIn]) -> list[ScoreOut]:
        art = request.app.state.art
        df = _to_frame(clients)
        out = score_clients(art["model"], df, art["params"], art["explainer"])
        version = art["metadata"].get("versao", "desconhecida")
        records = []
        for row in out.to_dict(orient="records"):
            row["id"] = None if pd.isna(row.get("id")) else int(row["id"])
            records.append(ScoreOut(**row, versao_modelo=version, versao_api=API_VERSION))
        return records

    @app.get("/health")
    def health() -> dict[str, str]:
        """Verificação de vida (usada pelo healthcheck do Docker)."""
        return {"status": "ok"}

    @app.get("/model-info", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        """Versão, métricas do holdout, cortes das faixas e premissas de negócio."""
        art = request.app.state.art
        meta = art["metadata"]
        return ModelInfo(
            versao_modelo=meta.get("versao", "desconhecida"),
            versao_api=API_VERSION,
            treinado_em=meta.get("treinado_em"),
            modelo=meta.get("modelo", config.FINAL_MODEL),
            variaveis_excluidas=meta.get("variaveis_excluidas", config.MODEL_EXCLUDE),
            metricas_holdout=meta.get("metricas_holdout", {}),
            cortes_faixas=art["params"].band_cuts,
            corte_valor_esperado=art["params"].ev_cut,
            premissas={
                "lgd": config.LGD,
                "eficacia_acao": config.ACTION_EFFICACY,
                "custo_acao": config.ACTION_COST,
                "custo_atrito": config.FRICTION_COST,
                "capacidade": config.CAPACITY,
            },
        )

    @app.post("/score", response_model=ScoreOut)
    def score(request: Request, client: ClientIn) -> ScoreOut:
        """Pontua um cliente."""
        return _score(request, [client])[0]

    @app.post("/score/batch", response_model=BatchOut)
    def score_batch(request: Request, batch: BatchIn) -> BatchOut:
        """Pontua até 1.000 clientes de uma vez."""
        results = _score(request, batch.clientes)
        return BatchOut(n=len(results), resultados=results)

    return app


app = create_app()

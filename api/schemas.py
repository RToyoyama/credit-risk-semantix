"""Contratos de entrada e saída da API (pydantic v2).

A entrada **não** aceita sexo, estado civil nem idade: o modelo não usa esses
dados (Etapa 13A) e a LGPD pede coletar só o necessário. Campos extras são
recusados com erro 422.
"""

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE = {
    "id": 101,
    "limit_bal": 50000,
    "education": 2,
    "pay_status_1": 2,
    "pay_status_2": 2,
    "pay_status_3": 0,
    "pay_status_4": 0,
    "pay_status_5": 0,
    "pay_status_6": 0,
    "bill_amt_1": 48500,
    "bill_amt_2": 47000,
    "bill_amt_3": 46200,
    "bill_amt_4": 30000,
    "bill_amt_5": 28000,
    "bill_amt_6": 25000,
    "pay_amt_1": 0,
    "pay_amt_2": 1500,
    "pay_amt_3": 2000,
    "pay_amt_4": 1500,
    "pay_amt_5": 1000,
    "pay_amt_6": 1000,
}

Status = Field(ge=-2, le=9, description="-2/-1 = pagou; 0 = rotativo; 1-9 = meses de atraso")
Bill = Field(description="Fatura do mês (NT$; pode ser negativa: crédito a favor)")
Paid = Field(ge=0, description="Valor pago no mês (NT$)")


class ClientIn(BaseModel):
    """Dados de um cliente. Índice 1 = mês mais recente."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE]})

    id: int | None = Field(default=None, description="Identificador do cliente (opcional)")
    limit_bal: float = Field(gt=0, description="Limite de crédito (NT$)")
    education: int = Field(
        ge=1, le=4, description="1 = pós-graduação, 2 = universidade, 3 = ensino médio, 4 = outros"
    )
    pay_status_1: int = Status
    pay_status_2: int = Status
    pay_status_3: int = Status
    pay_status_4: int = Status
    pay_status_5: int = Status
    pay_status_6: int = Status
    bill_amt_1: float = Bill
    bill_amt_2: float = Bill
    bill_amt_3: float = Bill
    bill_amt_4: float = Bill
    bill_amt_5: float = Bill
    bill_amt_6: float = Bill
    pay_amt_1: float = Paid
    pay_amt_2: float = Paid
    pay_amt_3: float = Paid
    pay_amt_4: float = Paid
    pay_amt_5: float = Paid
    pay_amt_6: float = Paid


class ScoreOut(BaseModel):
    """Resultado do scoring de um cliente."""

    id: int | None = None
    pd: float = Field(description="Probabilidade de calote no próximo mês (0 a 1)")
    ead: float = Field(description="Exposição no calote: fatura mais recente, negativa = 0 (NT$)")
    perda_esperada: float = Field(description="PD x LGD x EAD (NT$)")
    score: int = Field(
        description="0 a 1000; maior = menor risco; +60 pontos = chance de pagar dobra"
    )
    faixa: str = Field(description="A (menor risco) a E (maior risco)")
    acao: str = Field(description="Ação sugerida para a faixa")
    prioritario: bool = Field(description="Valor esperado de agir acima do corte da capacidade")
    valor_esperado_acao: float = Field(description="Valor esperado de acionar o cliente (NT$)")
    motivos: list[str] = Field(description="Até 3 fatores que mais aumentaram o risco")
    versao_modelo: str
    versao_api: str


class BatchIn(BaseModel):
    """Lote de clientes."""

    model_config = ConfigDict(extra="forbid")

    clientes: list[ClientIn] = Field(min_length=1, max_length=1000)


class BatchOut(BaseModel):
    """Resultado de um lote."""

    n: int
    resultados: list[ScoreOut]


class ModelInfo(BaseModel):
    """Metadados do modelo em produção."""

    versao_modelo: str
    versao_api: str
    treinado_em: str | None
    modelo: str
    variaveis_excluidas: list[str]
    metricas_holdout: dict[str, float]
    cortes_faixas: dict[str, float]
    corte_valor_esperado: float
    premissas: dict[str, float]

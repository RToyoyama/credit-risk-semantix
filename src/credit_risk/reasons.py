"""Motivos da decisão (*reason codes*): os fatores que mais aumentam o risco de cada cliente.

Usa TreeSHAP (exato para árvores) no modelo final, calculado pelo próprio
XGBoost (``pred_contribs=True``): é o mesmo algoritmo do ``shap.TreeExplainer``
(os testes conferem que os valores são iguais), mas devolve o valor base exato
que o modelo usa, o que evita uma incompatibilidade conhecida entre versões do
``shap`` e do XGBoost 3 na leitura do ``base_score``. As
contribuições são somadas por **conceito de negócio** antes de escolher os
motivos: sem isso, variáveis colineares (vários "atrasos") ocupariam os três
lugares. As contribuições estão em log-odds: somadas ao valor base, dão o
log-odds previsto do cliente.
"""

import pandas as pd
import xgboost as xgb

from credit_risk import config
from credit_risk.features import engineered_features


def concept_of(column: str) -> str:
    """Conceito de negócio de uma coluna do modelo.

    Args:
        column: Nome da coluna após o pré-processamento.

    Returns:
        Nome do conceito.

    Raises:
        KeyError: Se a coluna não pertencer a nenhum conceito.
    """
    for concept, cols in config.REASON_CONCEPTS.items():
        if column in cols or any(column.startswith(f"{c}_") for c in cols if c == "education"):
            return concept
    raise KeyError(f"Coluna sem conceito de negócio: {column}")


def _fmt_money(value: float) -> str:
    if abs(value) < 1_000:
        return f"NT$ {value:,.0f}".replace(",", ".")
    return f"NT$ {value / 1000:,.0f} mil".replace(",", ".")


def reason_phrase(concept: str, row: pd.Series) -> str:
    """Frase de negócio para um conceito, com os valores reais do cliente.

    Args:
        concept: Conceito de negócio.
        row: Dados limpos do cliente + features de resumo.

    Returns:
        Frase em português.
    """
    if concept == "atraso recente":
        months = int(max(row["pay_status_1"], 0))
        if months == 0:
            return "situação do último pagamento"
        return f"{months} {'mês' if months == 1 else 'meses'} de atraso no último pagamento"
    if concept == "histórico de atraso":
        k = int(row["delay_months"])
        if k == 0:
            return "histórico recente de pagamentos"
        return f"atraso em {k} dos últimos 6 meses (pior atraso: {int(row['delay_max'])} meses)"
    if concept == "uso do limite":
        if row["util_recent"] < 0.05:
            return f"cartão quase sem uso (usa {row['util_recent']:.0%} do limite)"
        return f"usa {row['util_recent']:.0%} do limite"
    if concept == "pagamentos":
        return f"comportamento de pagamento: pagou {row['pay_ratio_recent']:.0%} da fatura anterior"
    if concept == "limite de crédito":
        return f"limite de crédito de {_fmt_money(row['limit_bal'])}"
    if concept == "valor da fatura":
        if row["bill_amt_1"] <= 0:
            return "fatura zerada em setembro (cartão sem uso recente)"
        return f"fatura de {_fmt_money(row['bill_amt_1'])}"
    return "perfil cadastral (escolaridade)"


class ReasonExplainer:
    """Calcula contribuições SHAP e motivos para o pipeline final ajustado.

    Args:
        model: Pipeline final ajustado (``prep`` + ``model`` de árvores).
    """

    def __init__(self, model: object) -> None:
        self.prep = model.named_steps["prep"]
        self.booster = model.named_steps["model"].get_booster()
        self.base_value: float | None = None

    def contributions(self, clients: pd.DataFrame) -> pd.DataFrame:
        """Contribuição SHAP (log-odds) de cada coluna do modelo, por cliente.

        Args:
            clients: Dados limpos.

        Returns:
            Uma linha por cliente, uma coluna por variável do modelo.
        """
        Xt = self.prep.transform(clients)
        raw = self.booster.predict(xgb.DMatrix(Xt), pred_contribs=True)
        self.base_value = float(raw[0, -1])
        return pd.DataFrame(raw[:, :-1], columns=Xt.columns, index=clients.index)

    def concept_contributions(self, clients: pd.DataFrame) -> pd.DataFrame:
        """Contribuições somadas por conceito de negócio.

        Args:
            clients: Dados limpos.

        Returns:
            Uma linha por cliente, uma coluna por conceito.
        """
        contrib = self.contributions(clients)
        return contrib.T.groupby(contrib.columns.map(concept_of)).sum().T

    def explain(
        self,
        clients: pd.DataFrame,
        n: int = config.N_REASONS,
        min_contribution: float = config.MIN_REASON_CONTRIBUTION,
    ) -> list[list[dict]]:
        """Até ``n`` motivos por cliente, do maior para o menor.

        Só viram motivo os conceitos cuja contribuição é pelo menos
        ``min_contribution`` (em log-odds): 0,05 significa multiplicar a chance de
        calote por cerca de 1,05. Fatores menores são ruído para o cliente.

        Args:
            clients: Dados limpos.
            n: Número máximo de motivos.
            min_contribution: Contribuição mínima para virar motivo.

        Returns:
            Para cada cliente, lista de ``{"conceito", "contribuicao", "frase"}``;
            lista vazia se nenhum fator relevante aumenta o risco.
        """
        concepts = self.concept_contributions(clients)
        details = pd.concat([clients, engineered_features(clients)], axis=1)
        out = []
        for idx, row in concepts.iterrows():
            positive = row[row >= min_contribution].sort_values(ascending=False).head(n)
            out.append(
                [
                    {
                        "conceito": c,
                        "contribuicao": float(v),
                        "frase": reason_phrase(c, details.loc[idx]),
                    }
                    for c, v in positive.items()
                ]
            )
        return out

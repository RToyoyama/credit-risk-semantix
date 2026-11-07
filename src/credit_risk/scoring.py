"""Fonte única do scoring: PD → perda esperada → score → faixa → ação.

API, dashboard e avaliação usam estas funções; nenhuma regra de negócio é
repetida em outro lugar. Os cortes (faixas e valor esperado mínimo para ação)
são definidos com as previsões out-of-fold do treino e salvos em
``config.SCORING_PARAMS``.
"""

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from credit_risk import config
from credit_risk.decision import Premises, expected_value, exposure
from credit_risk.expected_loss import expected_loss

FACTOR = config.SCORE_PDO / np.log(2)
ANCHOR_ODDS = (1 - config.SCORE_ANCHOR_PD) / config.SCORE_ANCHOR_PD


def pd_to_score(pd_: np.ndarray | float) -> np.ndarray:
    """Converte PD em score de 0 a 1000 (maior = menor risco).

    score = âncora + (PDO / ln 2) × ln(chance de pagar ÷ chance na âncora).

    Args:
        pd_: Probabilidade de default.

    Returns:
        Score inteiro, limitado a ``config.SCORE_RANGE``.
    """
    p = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
    raw = config.SCORE_ANCHOR + FACTOR * np.log(((1 - p) / p) / ANCHOR_ODDS)
    return np.clip(np.round(raw), *config.SCORE_RANGE).astype(int)


def score_to_pd(score: np.ndarray | float) -> np.ndarray:
    """Inverso de ``pd_to_score`` (sem o arredondamento).

    Args:
        score: Score de 0 a 1000.

    Returns:
        PD correspondente.
    """
    odds = ANCHOR_ODDS * np.exp((np.asarray(score, dtype=float) - config.SCORE_ANCHOR) / FACTOR)
    return 1 / (1 + odds)


@dataclass(frozen=True)
class ScoringParams:
    """Cortes aprendidos no OOF do treino.

    Attributes:
        band_cuts: Score mínimo de cada faixa (A, B, C, D); E é o restante.
        ev_cut: Valor esperado mínimo para ser prioritário (equivale à capacidade).
    """

    band_cuts: dict[str, float]
    ev_cut: float


def fit_scoring_params(
    pd_oof: np.ndarray, ead: np.ndarray, prem: Premises | None = None
) -> ScoringParams:
    """Define os cortes das faixas e o valor esperado de corte com o OOF do treino.

    Args:
        pd_oof: PD out-of-fold dos clientes de treino.
        ead: Exposição dos mesmos clientes.
        prem: Premissas de negócio.

    Returns:
        Parâmetros do scoring.
    """
    prem = prem or Premises()
    scores = pd_to_score(pd_oof)
    cuts, cum = {}, 0.0
    for band, share in list(config.BAND_SHARES.items())[:-1]:
        cum += share
        cuts[band] = float(np.quantile(scores, 1 - cum))
    ev = expected_value(pd_oof, ead, prem)
    ev_cut = float(np.quantile(ev, 1 - config.CAPACITY))
    return ScoringParams(band_cuts=cuts, ev_cut=max(ev_cut, 0.0))


def save_params(params: ScoringParams) -> None:
    """Salva os cortes em ``config.SCORING_PARAMS``.

    Args:
        params: Parâmetros do scoring.
    """
    config.SCORING_PARAMS.write_text(json.dumps(asdict(params), indent=2) + "\n")


def load_params() -> ScoringParams:
    """Lê os cortes salvos.

    Returns:
        Parâmetros do scoring.

    Raises:
        FileNotFoundError: Se os cortes ainda não foram definidos.
    """
    if not config.SCORING_PARAMS.exists():
        raise FileNotFoundError(f"{config.SCORING_PARAMS} não existe. Rode `make experiments`.")
    return ScoringParams(**json.loads(config.SCORING_PARAMS.read_text()))


def assign_band(score: np.ndarray, cuts: dict[str, float]) -> np.ndarray:
    """Faixa A–E de cada score.

    Args:
        score: Score de 0 a 1000.
        cuts: Score mínimo de A, B, C e D.

    Returns:
        Vetor de letras.
    """
    score = np.asarray(score, dtype=float)
    bands = np.full(score.shape, list(config.BAND_SHARES)[-1], dtype=object)
    for band in reversed(list(cuts)):
        bands[score >= cuts[band]] = band
    return bands


def score_from_pd(
    pd_: np.ndarray, ead: np.ndarray, params: ScoringParams, prem: Premises | None = None
) -> pd.DataFrame:
    """Aplica toda a cadeia a partir da PD e da exposição.

    Args:
        pd_: Probabilidade de default.
        ead: Exposição (NT$).
        params: Cortes do scoring.
        prem: Premissas de negócio.

    Returns:
        Colunas: pd, ead, perda_esperada, score, faixa, acao, valor_esperado_acao, prioritario.
    """
    prem = prem or Premises()
    pd_ = np.asarray(pd_, dtype=float)
    ead = np.asarray(ead, dtype=float)
    score = pd_to_score(pd_)
    band = assign_band(score, params.band_cuts)
    ev = expected_value(pd_, ead, prem)
    return pd.DataFrame(
        {
            "pd": pd_,
            "ead": ead,
            "perda_esperada": expected_loss(pd_, ead, prem.lgd),
            "score": score,
            "faixa": band,
            "acao": [config.BAND_ACTIONS[b] for b in band],
            "valor_esperado_acao": ev,
            "prioritario": ev >= params.ev_cut,
        }
    )


def score_clients(
    model: object,
    clients: pd.DataFrame,
    params: ScoringParams,
    explainer: object | None = None,
) -> pd.DataFrame:
    """Scoring completo de clientes a partir dos dados limpos (uso da API e do dashboard).

    Args:
        model: Pipeline final ajustado.
        clients: Dados limpos (mesmo formato do treino, sem o alvo).
        params: Cortes do scoring.
        explainer: ``reasons.ReasonExplainer``; se informado, inclui os motivos.

    Returns:
        Resultado de ``score_from_pd`` com o ``id`` (se houver) e, opcionalmente,
        a coluna ``motivos`` (lista de frases).
    """
    p = model.predict_proba(clients)[:, 1]
    out = score_from_pd(p, exposure(clients).to_numpy(), params)
    if "id" in clients:
        out.insert(0, "id", clients["id"].to_numpy())
    if explainer is not None:
        reasons = explainer.explain(clients)
        out["motivos"] = [[r["frase"] for r in rs] or [config.NO_REASON_TEXT] for rs in reasons]
    return out

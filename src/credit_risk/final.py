"""Treino final e avaliação **única** no holdout (Etapa 13B).

O holdout fica lacrado desde a Etapa 4. Este módulo o abre uma vez: treina o
modelo oficial no treino completo, prevê o holdout e salva tudo. Se as
métricas do holdout já existirem, ele se recusa a rodar de novo (``force``
existe só para reprodução, e o uso fica registrado no log).

Uso: ``python -m credit_risk.final`` (ou ``make final``).
"""

import json
import logging
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from credit_risk import __version__, config
from credit_risk.decision import (
    Premises,
    bootstrap_diff,
    capacity_mask,
    expected_value,
    exposure,
    policy_result,
    realized_value,
    rule_priority,
)
from credit_risk.evaluate import ks_statistic
from credit_risk.models import final_pipeline
from credit_risk.segmentation import SegmentEncoder
from credit_risk.split import load_train

logger = logging.getLogger(__name__)

METRICS = {
    "pr_auc": average_precision_score,
    "roc_auc": roc_auc_score,
    "ks": ks_statistic,
    "brier": brier_score_loss,
}


class HoldoutAlreadyUsedError(RuntimeError):
    """O holdout já foi avaliado; reavaliar quebraria a regra de uso único."""


def metrics_with_ci(
    y: np.ndarray, p: np.ndarray, n_boot: int = config.N_BOOT, seed: int = config.SEED
) -> pd.DataFrame:
    """Métricas com IC 95% por bootstrap (reamostrando clientes).

    Args:
        y: Alvo real.
        p: Probabilidade prevista.
        n_boot: Reamostragens.
        seed: Semente.

    Returns:
        Uma linha por métrica: valor, ic_inf, ic_sup.
    """
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    rows = []
    for name, fn in METRICS.items():
        boots = np.array([fn(y[i], p[i]) for i in idx])
        low, high = np.percentile(boots, [2.5, 97.5])
        rows.append(
            {"métrica": name, "valor": float(fn(y, p)), "ic_inf": float(low), "ic_sup": float(high)}
        )
    return pd.DataFrame(rows)


def group_report(y: np.ndarray, p: np.ndarray, act: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    """Desempenho, calibração e taxa de acionamento por grupo.

    Args:
        y: Alvo real.
        p: Probabilidade prevista.
        act: Máscara de clientes acionados pela política.
        groups: Grupo de cada cliente.

    Returns:
        Uma linha por grupo.
    """
    df = pd.DataFrame(
        {"y": np.asarray(y), "p": np.asarray(p), "act": np.asarray(act), "g": groups.to_numpy()}
    )
    rows = []
    for g, part in df.groupby("g", observed=True):
        rows.append(
            {
                "grupo": g,
                "clientes": len(part),
                "taxa_real": part["y"].mean(),
                "pd_media": part["p"].mean(),
                "gap_calibracao": part["p"].mean() - part["y"].mean(),
                "roc_auc": roc_auc_score(part["y"], part["p"]),
                "pr_auc": average_precision_score(part["y"], part["p"]),
                "pct_acionados": part["act"].mean(),
                "precisao_acionados": part.loc[part["act"], "y"].mean()
                if part["act"].any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def partial_dependence(
    model: object, X: pd.DataFrame, column: str, grid: list[float]
) -> pd.DataFrame:
    """PD média quando ``column`` assume cada valor da grade, para todos os clientes.

    Args:
        model: Pipeline ajustado (recebe dados limpos).
        X: Dados limpos.
        column: Coluna variada.
        grid: Valores testados.

    Returns:
        Uma linha por valor: ``valor`` e ``pd_media``.
    """
    rows = []
    for v in grid:
        Xv = X.copy()
        Xv[column] = v
        rows.append(
            {
                "variável": column,
                "valor": v,
                "pd_media": float(model.predict_proba(Xv)[:, 1].mean()),
            }
        )
    return pd.DataFrame(rows)


def _cv_reference() -> pd.DataFrame:
    """Faixa da CV do modelo final (15 folds), para comparar com o holdout."""
    from credit_risk.evaluate import cv_scores

    train = load_train()
    folds = cv_scores(final_pipeline(), train.drop(columns=[config.TARGET]), train[config.TARGET])
    out = folds.agg(["mean", "std"]).T.reset_index(names="métrica")
    out["faixa_inf"] = out["mean"] - 1.96 * out["std"]
    out["faixa_sup"] = out["mean"] + 1.96 * out["std"]
    return out.rename(columns={"mean": "cv_media", "std": "cv_desvio"})


def run_final(force: bool = False) -> pd.DataFrame:
    """Treina o modelo oficial no treino completo e avalia o holdout uma única vez.

    Args:
        force: Reavalia mesmo que o holdout já tenha sido usado (só para reprodução).

    Returns:
        Métricas do holdout com IC e a faixa da CV.

    Raises:
        HoldoutAlreadyUsedError: Se o holdout já foi avaliado e ``force`` é False.
    """
    if config.HOLDOUT_METRICS.exists() and not force:
        raise HoldoutAlreadyUsedError(
            f"{config.HOLDOUT_METRICS} já existe: o holdout já foi usado. "
            "Use force=True só para reproduzir."
        )
    if force:
        logger.warning("Holdout reavaliado com force=True (reprodução).")

    train = load_train()
    X_train, y_train = train.drop(columns=[config.TARGET]), train[config.TARGET]
    model = final_pipeline().fit(X_train, y_train)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.FINAL_MODEL_PATH)

    holdout = pd.read_parquet(config.HOLDOUT_PARQUET)
    X_ho, y_ho = holdout.drop(columns=[config.TARGET]), holdout[config.TARGET].to_numpy()
    p = model.predict_proba(X_ho)[:, 1]
    ead = exposure(holdout).to_numpy()
    segments = SegmentEncoder().fit(train).predict(holdout)
    pd.DataFrame(
        {"id": holdout["id"], config.TARGET: y_ho, "ead": ead, "pd": p, "segment": segments}
    ).to_parquet(config.HOLDOUT_PREDICTIONS, index=False)

    metrics = metrics_with_ci(y_ho, p).merge(_cv_reference(), on="métrica", how="left")
    metrics["dentro_da_faixa_cv"] = (metrics["valor"] >= metrics["faixa_inf"]) & (
        metrics["valor"] <= metrics["faixa_sup"]
    )

    prem = Premises()
    ev = expected_value(p, ead, prem)
    masks = {
        "maior valor esperado (modelo)": capacity_mask(ev, config.CAPACITY, ev > 0),
        "maior PD (modelo)": capacity_mask(p, config.CAPACITY),
        "regra de atraso (sem modelo)": capacity_mask(rule_priority(holdout), config.CAPACITY),
    }
    policy = pd.DataFrame(
        [{"política": n, **policy_result(m, y_ho, ead, prem)} for n, m in masks.items()]
    )
    value = realized_value(y_ho, ead, prem)
    v = {n: np.where(m, value, 0.0) for n, m in masks.items()}
    diffs = [
        (
            "valor esperado − regra de atraso",
            *bootstrap_diff(v["maior valor esperado (modelo)"], v["regra de atraso (sem modelo)"]),
        ),
        (
            "maior PD − regra de atraso",
            *bootstrap_diff(v["maior PD (modelo)"], v["regra de atraso (sem modelo)"]),
        ),
    ]
    act = masks["maior valor esperado (modelo)"]
    confusion = pd.crosstab(
        pd.Series(np.where(act, "acionado", "não acionado"), name="decisão"),
        pd.Series(np.where(y_ho == 1, "deu calote", "pagou"), name="realidade"),
    )

    age_band = pd.cut(holdout["age"], config.AGE_BANDS, labels=["21–30", "31–45", "46+"])
    sex = holdout["sex"].map(config.CATEGORY_LABELS["sex"])
    seg_names = pd.Series(segments).map(config.SEGMENT_NAMES)
    fairness = pd.concat(
        [
            group_report(y_ho, p, act, sex).assign(atributo="sexo"),
            group_report(y_ho, p, act, age_band).assign(atributo="faixa de idade"),
        ],
        ignore_index=True,
    )
    by_segment = group_report(y_ho, p, act, seg_names)

    perm = permutation_importance(
        model,
        X_ho,
        y_ho,
        scoring="average_precision",
        n_repeats=config.PERMUTATION_REPEATS,
        random_state=config.SEED,
        n_jobs=-1,
    )
    importance = (
        pd.DataFrame(
            {
                "variável": X_ho.columns,
                "queda_pr_auc": perm.importances_mean,
                "desvio": perm.importances_std,
            }
        )
        .sort_values("queda_pr_auc", ascending=False)
        .reset_index(drop=True)
    )
    pdp = pd.concat(
        [
            partial_dependence(model, X_ho, "pay_status_1", [-2, -1, 0, 1, 2, 3, 4]),
            partial_dependence(
                model, X_ho, "limit_bal", [20_000, 50_000, 100_000, 200_000, 300_000, 500_000]
            ),
        ],
        ignore_index=True,
    )

    out = config.REPORTS_DIR
    policy.to_csv(out / "holdout_policy.csv", index=False)
    pd.DataFrame(diffs, columns=["comparação", "diferença", "ic_inf", "ic_sup"]).to_csv(
        out / "holdout_policy_bootstrap.csv", index=False
    )
    confusion.to_csv(out / "holdout_confusion.csv")
    fairness.to_csv(out / "holdout_fairness.csv", index=False)
    by_segment.to_csv(out / "holdout_by_segment.csv", index=False)
    importance.to_csv(out / "permutation_importance.csv", index=False)
    pdp.to_csv(out / "partial_dependence.csv", index=False)
    metrics.to_csv(config.HOLDOUT_METRICS, index=False)

    metadata = {
        "versao": __version__,
        "treinado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "modelo": config.FINAL_MODEL,
        "variaveis_excluidas": config.MODEL_EXCLUDE,
        "clientes_treino": len(train),
        "metricas_holdout": metrics.set_index("métrica")["valor"].round(4).to_dict(),
    }
    config.MODEL_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    logger.info("Holdout avaliado e lacrado: %s", config.HOLDOUT_METRICS)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print(run_final().round(4).to_string(index=False))

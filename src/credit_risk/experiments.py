"""Experimentos pareados do projeto, cada um salvo em ``reports/``.

Cada experimento compara variantes nos mesmos folds (CV 5 x 3) e aplica
o critério fixado antes de rodar: adotar só se o PR AUC subir com win
rate > ``config.WIN_RATE_MIN``.

Uso: ``python -m credit_risk.experiments`` (ou ``make experiments``).
"""

import logging

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from credit_risk import config
from credit_risk.balancing import balanced_pipeline
from credit_risk.decision import calibrated_pipeline
from credit_risk.evaluate import (
    cv_scores,
    oof_predictions,
    paired_comparison,
    reliability_table,
    summarize,
)
from credit_risk.models import CATALOG, ENSEMBLES, build_ensemble, build_pipeline
from credit_risk.preprocessing import make_feature_pipeline
from credit_risk.split import load_train

logger = logging.getLogger(__name__)

METRICS = ["pr_auc", "roc_auc", "brier"]
BALANCING_MODELS = ["xgboost", "random_forest", "logistic"]


def reference_models() -> dict[str, BaseEstimator]:
    """Modelos de referência dos experimentos de features (do catálogo, sem tuning).

    Returns:
        Regressão logística e XGBoost.
    """
    return {"logística": CATALOG["logistic"].factory(), "xgboost": CATALOG["xgboost"].factory()}


def model_pipeline(
    model: BaseEstimator,
    scaled: bool = True,
    engineered: bool = True,
    status_as_category: bool = False,
) -> Pipeline:
    """Pipeline completo: features + pré-processamento oficial + modelo.

    Args:
        model: Classificador.
        scaled: Padroniza os números.
        engineered: Inclui as features de resumo.
        status_as_category: Trata ``pay_status_*`` como categoria.

    Returns:
        Pipeline não ajustado.
    """
    return Pipeline(
        [
            ("prep", make_feature_pipeline(scaled, engineered, status_as_category)),
            ("model", model),
        ]
    )


def _compare(variants: dict[str, tuple[dict, dict]], name: str) -> pd.DataFrame:
    """Roda pares (base, candidato) de configurações e salva a comparação."""
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    models = reference_models()
    rows = []
    for model_name, (base_kw, cand_kw) in variants.items():
        model = models[model_name]
        base = cv_scores(model_pipeline(model, **base_kw), X, y)
        cand = cv_scores(model_pipeline(model, **cand_kw), X, y)
        for metric in METRICS:
            rows.append(
                {"modelo": model_name, "métrica": metric, **paired_comparison(base, cand, metric)}
            )
        logger.info("%s: %s concluído", name, model_name)
    out = decide(pd.DataFrame(rows))
    out.to_csv(config.REPORTS_DIR / f"{name}.csv", index=False)
    return out


def decide(results: pd.DataFrame) -> pd.DataFrame:
    """Marca a coluna ``adotar`` segundo o critério fixado (PR AUC + win rate).

    Args:
        results: Linhas de ``paired_comparison`` com colunas ``modelo`` e ``métrica``.

    Returns:
        A mesma tabela com a coluna ``adotar``.
    """
    is_pr = results["métrica"] == "pr_auc"
    results["adotar"] = (
        is_pr & (results["ganho_medio"] > 0) & (results["win_rate"] > config.WIN_RATE_MIN)
    )
    return results


def feature_experiment() -> pd.DataFrame:
    """Etapa 7: colunas originais × originais + features de resumo.

    Returns:
        Comparação pareada por modelo e métrica (salva em ``reports/``).
    """
    return _compare(
        {
            "logística": ({"engineered": False}, {"engineered": True}),
            "xgboost": ({"scaled": False, "engineered": False}, {"scaled": False}),
        },
        "feature_experiment",
    )


def status_category_experiment() -> pd.DataFrame:
    """Etapa 8: ``pay_status`` como número × como categoria, na logística.

    Árvores ficam de fora: elas cortam em qualquer ponto, então tratar o
    status como número não as engana.

    Returns:
        Comparação pareada (salva em ``reports/``).
    """
    return _compare(
        {"logística": ({}, {"status_as_category": True})},
        "status_category_experiment",
    )


def model_duel() -> pd.DataFrame:
    """Etapa 9: os quatro modelos (e a referência) nos mesmos 15 folds.

    Salva as métricas de cada fold (``model_duel_folds.csv``) e o resumo com a
    comparação pareada contra o líder em PR AUC (``model_duel_summary.csv``).

    Returns:
        Resumo ordenado por PR AUC.
    """
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    folds = {}
    for name in CATALOG:
        folds[name] = cv_scores(build_pipeline(name), X, y)
        logger.info("Duelo: %s concluído", name)
    long = pd.concat(
        [s.assign(modelo=n, fold=range(len(s))) for n, s in folds.items()], ignore_index=True
    )
    long.to_csv(config.REPORTS_DIR / "model_duel_folds.csv", index=False)
    leader = max((n for n in folds if n != "dummy"), key=lambda n: folds[n]["pr_auc"].mean())
    summary = summarize(folds, reference=leader)
    summary.insert(1, "nome", summary["modelo"].map(lambda n: CATALOG[n].label))
    summary.to_csv(config.REPORTS_DIR / "model_duel_summary.csv", index=False)
    return summary


def decide_with_calibration(results: pd.DataFrame) -> pd.DataFrame:
    """Critério com veto da calibração: PR AUC sobe (win rate > 60%) e Brier não piora.

    Args:
        results: Comparações pareadas com colunas ``modelo``, ``estrategia``,
            ``métrica``, ``ganho_medio`` e ``win_rate``.

    Returns:
        Uma linha por modelo e estratégia, com a decisão.
    """
    wide = results.pivot_table(
        index=["modelo", "estrategia"], columns="métrica", values=["ganho_medio", "win_rate"]
    )
    out = pd.DataFrame(
        {
            "ganho_pr_auc": wide[("ganho_medio", "pr_auc")],
            "win_rate_pr_auc": wide[("win_rate", "pr_auc")],
            "ganho_brier": wide[("ganho_medio", "brier")],
            "win_rate_brier": wide[("win_rate", "brier")],
        }
    )
    out["pr_auc_ok"] = (out["ganho_pr_auc"] > 0) & (out["win_rate_pr_auc"] > config.WIN_RATE_MIN)
    out["brier_ok"] = out["win_rate_brier"] >= 0.5
    out["adotar"] = out["pr_auc_ok"] & out["brier_ok"]
    return out.reset_index()


def balancing_experiment() -> pd.DataFrame:
    """Etapa 10: nenhum × class_weight × SMOTE, nos mesmos folds.

    Salva as comparações (``balancing_experiment.csv``) e a decisão com veto
    da calibração (``balancing_decision.csv``).

    Returns:
        Tabela de decisão por modelo e estratégia.
    """
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    rows = []
    for name in BALANCING_MODELS:
        base = cv_scores(balanced_pipeline(name, "none"), X, y)
        for strategy in ("class_weight", "smote"):
            cand = cv_scores(balanced_pipeline(name, strategy), X, y)
            for metric in ["pr_auc", "roc_auc", "ks", "brier"]:
                rows.append(
                    {
                        "modelo": name,
                        "estrategia": strategy,
                        "métrica": metric,
                        **paired_comparison(base, cand, metric),
                    }
                )
            logger.info("Balanceamento: %s + %s concluído", name, strategy)
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(config.REPORTS_DIR / "balancing_experiment.csv", index=False)
    decision = decide_with_calibration(comparisons)
    decision.to_csv(config.REPORTS_DIR / "balancing_decision.csv", index=False)
    return decision


def balancing_calibration() -> pd.DataFrame:
    """Etapa 10: curva de confiabilidade (OOF) de cada modelo e estratégia.

    Returns:
        Tabela longa com 10 faixas por variante (salva em ``reports/``).
    """
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    tables = []
    for name in BALANCING_MODELS:
        for strategy in ["none", "class_weight", "smote"]:
            p = oof_predictions(balanced_pipeline(name, strategy), X, y)
            tables.append(
                reliability_table(y, p).assign(
                    modelo=name, estrategia=strategy, media_prevista=p.mean()
                )
            )
            logger.info("Calibração: %s + %s concluído", name, strategy)
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(config.REPORTS_DIR / "balancing_calibration.csv", index=False)
    return out


DIVERSITY_MODELS = ["xgboost", "random_forest", "logistic", "decision_tree"]


def ensemble_diversity() -> pd.DataFrame:
    """Etapa 11B: correlação entre os erros out-of-fold dos modelos ajustados.

    Erro = alvo − probabilidade prevista. Correlação perto de 1 significa que
    os modelos erram nos mesmos clientes, e combiná-los ajuda pouco.

    Returns:
        Matriz de correlação (salva em ``reports/ensemble_diversity.csv``).
    """
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    errors = pd.DataFrame(
        {
            m: y.to_numpy() - oof_predictions(build_pipeline(m, tuned=True), X, y)
            for m in DIVERSITY_MODELS
        }
    )
    corr = errors.corr()
    corr.to_csv(config.REPORTS_DIR / "ensemble_diversity.csv")
    return corr


def ensemble_experiment() -> pd.DataFrame:
    """Etapa 11B: cada ensemble contra o XGBoost ajustado, nos mesmos folds.

    Returns:
        Decisão com veto da calibração (salva em ``reports/``).
    """
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    base = cv_scores(build_pipeline("xgboost", tuned=True), X, y)
    rows = []
    for name in ENSEMBLES:
        cand = cv_scores(build_ensemble(name), X, y)
        for metric in ["pr_auc", "roc_auc", "ks", "brier"]:
            rows.append(
                {
                    "modelo": "xgboost",
                    "estrategia": name,
                    "métrica": metric,
                    **paired_comparison(base, cand, metric),
                }
            )
        logger.info("Ensemble: %s concluído", name)
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(config.REPORTS_DIR / "ensemble_experiment.csv", index=False)
    decision = decide_with_calibration(comparisons)
    decision.to_csv(config.REPORTS_DIR / "ensemble_decision.csv", index=False)
    return decision


def calibration_experiment() -> pd.DataFrame:
    """Etapa 12A: XGBoost ajustado sem correção × sigmoid × isotônica.

    Critério: adotar se o Brier melhorar com win rate > 60% e o PR AUC não
    piorar (win rate >= 50%). Salva também a curva de confiabilidade (OOF).

    Returns:
        Decisão por método (``reports/calibration_decision.csv``).
    """
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    base = cv_scores(calibrated_pipeline("none"), X, y)
    rows, tables = [], []
    for method in ["sigmoid", "isotonic"]:
        cand = cv_scores(calibrated_pipeline(method), X, y)
        for metric in ["brier", "pr_auc", "roc_auc", "ks"]:
            rows.append(
                {"método": method, "métrica": metric, **paired_comparison(base, cand, metric)}
            )
        logger.info("Calibração: %s concluído", method)
    for method in ["none", "sigmoid", "isotonic"]:
        p = oof_predictions(calibrated_pipeline(method), X, y)
        tables.append(reliability_table(y, p).assign(método=method, media_prevista=p.mean()))
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(config.REPORTS_DIR / "calibration_experiment.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(
        config.REPORTS_DIR / "calibration_reliability.csv", index=False
    )
    wide = comparisons.pivot_table(
        index="método", columns="métrica", values=["ganho_medio", "win_rate"]
    )
    decision = pd.DataFrame(
        {
            "ganho_brier": wide[("ganho_medio", "brier")],
            "win_rate_brier": wide[("win_rate", "brier")],
            "ganho_pr_auc": wide[("ganho_medio", "pr_auc")],
            "win_rate_pr_auc": wide[("win_rate", "pr_auc")],
        }
    )
    decision["adotar"] = (
        (decision["ganho_brier"] > 0)
        & (decision["win_rate_brier"] > config.WIN_RATE_MIN)
        & (decision["win_rate_pr_auc"] >= 0.5)
    )
    decision = decision.reset_index()
    decision.to_csv(config.REPORTS_DIR / "calibration_decision.csv", index=False)
    return decision


EXPERIMENTS = {
    "feature_experiment": feature_experiment,
    "status_category_experiment": status_category_experiment,
    "model_duel": model_duel,
    "balancing_experiment": balancing_experiment,
    "balancing_calibration": balancing_calibration,
    "ensemble_diversity": ensemble_diversity,
    "ensemble_experiment": ensemble_experiment,
    "calibration_experiment": calibration_experiment,
}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    chosen = sys.argv[1:] or list(EXPERIMENTS)
    for exp_name, run in ((n, EXPERIMENTS[n]) for n in chosen):
        print(f"\n== {exp_name} ==")
        print(run().round(4).to_string(index=False))

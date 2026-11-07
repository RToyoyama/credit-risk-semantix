"""Experimentos pareados do projeto, cada um salvo em ``reports/``.

Cada experimento compara variantes nos mesmos folds (CV 5 x 3) e aplica
o critério fixado antes de rodar: adotar só se o PR AUC subir com win
rate > ``config.WIN_RATE_MIN``.

Uso: ``python -m credit_risk.experiments`` (ou ``make experiments``).
"""

import logging
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from credit_risk import config
from credit_risk.balancing import balanced_pipeline
from credit_risk.decision import (
    Premises,
    bootstrap_diff,
    calibrated_pipeline,
    capacity_mask,
    delay_rules,
    expected_value,
    exposure,
    policy_result,
    realized_value,
    rule_priority,
    threshold_curve,
)
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


def _policies(
    p: np.ndarray, y: np.ndarray, ead: np.ndarray, df: pd.DataFrame, prem: Premises
) -> dict:
    """Máscaras de cada política comparada (limiar ótimo e regra fixa escolhidos no treino)."""
    curve = threshold_curve(p, y, ead, prem)
    best_t = float(curve.loc[curve["valor"].idxmax(), "limiar"])
    rules = delay_rules(df)
    best_rule = max(rules, key=lambda r: policy_result(rules[r], y, ead, prem)["valor"])
    return (
        {
            "não agir": np.zeros(len(y), dtype=bool),
            "agir em todos": np.ones(len(y), dtype=bool),
            "limiar 0,5": p >= 0.5,
            f"melhor limiar único ({best_t:.2f})": p >= best_t,
            "regra por cliente (valor esperado > 0)": expected_value(p, ead, prem) > 0,
            f"melhor regra fixa: {best_rule}": rules[best_rule],
        },
        curve,
        best_t,
        best_rule,
    )


def decision_experiment() -> pd.DataFrame:
    """Etapa 12B: políticas de decisão nas previsões OOF do treino, em NT$.

    Salva: previsões OOF (``data/processed``), tabela de políticas, curva de
    valor × limiar, ICs bootstrap das diferenças, sensibilidade às premissas e
    a checagem de robustez com a calibração isotônica.

    Returns:
        Tabela de políticas.
    """
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET].to_numpy()
    ead = exposure(df).to_numpy()
    p = oof_predictions(calibrated_pipeline("none"), X, df[config.TARGET])
    p_iso = oof_predictions(calibrated_pipeline("isotonic"), X, df[config.TARGET])
    pd.DataFrame(
        {"id": df["id"], config.TARGET: y, "ead": ead, "pd": p, "pd_isotonic": p_iso}
    ).to_parquet(config.OOF_PREDICTIONS, index=False)
    prem = Premises()
    masks, curve, best_t, best_rule = _policies(p, y, ead, df, prem)
    curve.to_csv(config.REPORTS_DIR / "decision_threshold_curve.csv", index=False)
    table = pd.DataFrame(
        [{"política": n, **policy_result(m, y, ead, prem)} for n, m in masks.items()]
    )
    table.to_csv(config.REPORTS_DIR / "decision_policies.csv", index=False)

    value = realized_value(y, ead, prem)
    per_client = {n: np.where(m, value, 0.0) for n, m in masks.items()}
    client = per_client["regra por cliente (valor esperado > 0)"]
    rule = per_client[f"melhor regra fixa: {best_rule}"]
    single = per_client[f"melhor limiar único ({best_t:.2f})"]
    cis = [
        ("regra por cliente − não agir", *bootstrap_diff(client, np.zeros_like(client))),
        ("regra por cliente − melhor regra fixa", *bootstrap_diff(client, rule)),
        ("regra por cliente − melhor limiar único", *bootstrap_diff(client, single)),
        ("melhor limiar único − melhor regra fixa", *bootstrap_diff(single, rule)),
    ]
    pd.DataFrame(cis, columns=["comparação", "diferença", "ic_inf", "ic_sup"]).to_csv(
        config.REPORTS_DIR / "decision_bootstrap.csv", index=False
    )

    rows = []
    for param, values in config.SENSITIVITY.items():
        for v in values:
            pr = replace(prem, **{param: v})
            m, _, t_star, r_name = _policies(p, y, ead, df, pr)
            res_client = policy_result(m["regra por cliente (valor esperado > 0)"], y, ead, pr)
            res_rule = policy_result(m[f"melhor regra fixa: {r_name}"], y, ead, pr)
            res_single = policy_result(m[f"melhor limiar único ({t_star:.2f})"], y, ead, pr)
            rows.append(
                {
                    "premissa": param,
                    "valor": v,
                    "valor_regra_cliente": res_client["valor"],
                    "pct_acionados_regra_cliente": res_client["pct_acionados"],
                    "melhor_limiar": t_star,
                    "valor_melhor_limiar": res_single["valor"],
                    "melhor_regra_fixa": r_name,
                    "valor_melhor_regra_fixa": res_rule["valor"],
                    "perda_sem_acao": res_client["perda_sem_acao"],
                }
            )
    pd.DataFrame(rows).to_csv(config.REPORTS_DIR / "decision_sensitivity.csv", index=False)

    robust = [
        {"probabilidade": name, **policy_result(expected_value(q, ead, prem) > 0, y, ead, prem)}
        for name, q in [("sem correção", p), ("isotônica", p_iso)]
    ]
    pd.DataFrame(robust).to_csv(config.REPORTS_DIR / "decision_robustness.csv", index=False)
    logger.info("Decisão concluída (melhor limiar %.2f; melhor regra: %s)", best_t, best_rule)
    return table


def capacity_experiment() -> pd.DataFrame:
    """Etapa 12B: quem acionar com capacidade limitada (mesmo número de ações).

    Compara, nas previsões OOF salvas por ``decision_experiment``: maior valor
    esperado (com valor esperado > 0), maior PD e a regra de atraso. Critério:
    valor esperado vence a regra de atraso com IC 95% acima de zero.

    Returns:
        Resultado por capacidade e política (``reports/decision_capacity.csv``).
    """
    df = load_train()
    oof = pd.read_parquet(config.OOF_PREDICTIONS).set_index("id").loc[df["id"]]
    y, ead, p = oof[config.TARGET].to_numpy(), oof["ead"].to_numpy(), oof["pd"].to_numpy()
    prem = Premises()
    ev = expected_value(p, ead, prem)
    value = realized_value(y, ead, prem)
    rows, cis = [], []
    for cap in config.CAPACITY_SENSITIVITY:
        masks = {
            "maior valor esperado (modelo)": capacity_mask(ev, cap, eligible=ev > 0),
            "maior PD (modelo)": capacity_mask(p, cap),
            "regra de atraso (sem modelo)": capacity_mask(rule_priority(df), cap),
        }
        for name, m in masks.items():
            rows.append({"capacidade": cap, "política": name, **policy_result(m, y, ead, prem)})
        v = {n: np.where(m, value, 0.0) for n, m in masks.items()}
        pairs = [
            ("maior valor esperado (modelo)", "regra de atraso (sem modelo)"),
            ("maior valor esperado (modelo)", "maior PD (modelo)"),
            ("maior PD (modelo)", "regra de atraso (sem modelo)"),
        ]
        for a, b in pairs:
            cis.append((cap, f"{a} − {b}", *bootstrap_diff(v[a], v[b])))
    out = pd.DataFrame(rows)
    out.to_csv(config.REPORTS_DIR / "decision_capacity.csv", index=False)
    rule_mask = capacity_mask(rule_priority(df), config.CAPACITY)
    sens = []
    for param, values in config.SENSITIVITY.items():
        for val in values:
            pr = replace(prem, **{param: val})
            ev_p = expected_value(p, ead, pr)
            res_ev = policy_result(capacity_mask(ev_p, config.CAPACITY, ev_p > 0), y, ead, pr)
            res_rule = policy_result(rule_mask, y, ead, pr)
            sens.append(
                {
                    "premissa": param,
                    "valor": val,
                    "valor_modelo": res_ev["valor"],
                    "valor_regra": res_rule["valor"],
                    "diferenca": res_ev["valor"] - res_rule["valor"],
                    "pct_perda_evitada_modelo": res_ev["valor"] / res_ev["perda_sem_acao"],
                }
            )
    pd.DataFrame(sens).to_csv(config.REPORTS_DIR / "decision_capacity_sensitivity.csv", index=False)
    pd.DataFrame(cis, columns=["capacidade", "comparação", "diferença", "ic_inf", "ic_sup"]).to_csv(
        config.REPORTS_DIR / "decision_capacity_bootstrap.csv", index=False
    )
    return out


def sensitive_experiment() -> pd.DataFrame:
    """Etapa 13A: remover atributos sensíveis (sexo, estado civil, idade)?

    Critério de não inferioridade, fixado antes: remover se a perda de PR AUC
    for menor que ``config.NONINFERIORITY_PR_AUC`` e a do valor da política
    (capacidade de 10%, OOF) for menor que ``config.NONINFERIORITY_VALUE``.
    Prefere-se a variante mais restritiva que passar.

    Returns:
        Uma linha por variante (``reports/sensitive_experiment.csv``).
    """
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    ead = exposure(df).to_numpy()
    prem = Premises()
    folds, values = {}, {}
    for name, excl in config.SENSITIVE_VARIANTS.items():
        pipe = build_pipeline(config.FINAL_MODEL, tuned=True, exclude=tuple(excl))
        folds[name] = cv_scores(pipe, X, y)
        p = oof_predictions(pipe, X, y)
        ev = expected_value(p, ead, prem)
        values[name] = policy_result(
            capacity_mask(ev, config.CAPACITY, ev > 0), y.to_numpy(), ead, prem
        )["valor"]
        logger.info("Sensíveis: %s concluído", name)
    rows = []
    for name in config.SENSITIVE_VARIANTS:
        comp = paired_comparison(folds["atual"], folds[name], "pr_auc")
        value_loss = 1 - values[name] / values["atual"]
        rows.append(
            {
                "variante": name,
                "pr_auc": comp["candidato"],
                "perda_pr_auc": -comp["ganho_medio"],
                "win_rate_vs_atual": comp["win_rate"],
                "valor_politica": values[name],
                "perda_valor_pct": value_loss,
                "nao_inferior": (-comp["ganho_medio"] < config.NONINFERIORITY_PR_AUC)
                and (value_loss < config.NONINFERIORITY_VALUE),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(config.REPORTS_DIR / "sensitive_experiment.csv", index=False)
    return out


EXPERIMENTS = {
    "feature_experiment": feature_experiment,
    "status_category_experiment": status_category_experiment,
    "model_duel": model_duel,
    "balancing_experiment": balancing_experiment,
    "balancing_calibration": balancing_calibration,
    "ensemble_diversity": ensemble_diversity,
    "ensemble_experiment": ensemble_experiment,
    "calibration_experiment": calibration_experiment,
    "decision_experiment": decision_experiment,
    "capacity_experiment": capacity_experiment,
    "sensitive_experiment": sensitive_experiment,
}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    chosen = sys.argv[1:] or list(EXPERIMENTS)
    for exp_name, run in ((n, EXPERIMENTS[n]) for n in chosen):
        print(f"\n== {exp_name} ==")
        print(run().round(4).to_string(index=False))

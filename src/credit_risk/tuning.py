"""Ajuste de hiperparâmetros com semente de busca separada.

A busca usa folds sorteados com ``config.SEARCH_SEED``; a avaliação usa os
folds padrão (seed 42). A diferença entre a nota que a busca reportou e a
nota nos folds padrão mede o **otimismo da busca**: escolher o melhor entre
muitos candidatos sempre favorece quem teve sorte naqueles folds.

Uso: ``python -m credit_risk.tuning`` (ou ``make tune``).
"""

import json
import logging
from typing import Any

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from credit_risk import config
from credit_risk.evaluate import cv_scores, paired_comparison
from credit_risk.models import CATALOG, build_pipeline
from credit_risk.split import load_train

logger = logging.getLogger(__name__)

GRIDS: dict[str, dict[str, list]] = {
    "logistic": {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
    "decision_tree": {
        "model__max_depth": [3, 4, 5, 6, 8, 10],
        "model__min_samples_leaf": [20, 50, 100, 200],
    },
}

DISTRIBUTIONS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 8, 12, 16],
        "model__min_samples_leaf": [5, 10, 20, 40],
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "xgboost": {
        "model__n_estimators": randint(200, 801),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(2, 7),
        "model__min_child_weight": loguniform(1, 20),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_lambda": loguniform(0.1, 10),
    },
}

TUNABLE = ["logistic", "decision_tree", "random_forest", "xgboost"]


def make_search(name: str) -> GridSearchCV | RandomizedSearchCV:
    """Busca de hiperparâmetros de um modelo (Grid ou Randomized).

    Args:
        name: Chave em ``models.CATALOG``.

    Returns:
        Objeto de busca não ajustado, otimizando PR AUC.
    """
    cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.SEARCH_SEED)
    common = {"scoring": "average_precision", "cv": cv, "n_jobs": -1, "refit": False}
    if name in GRIDS:
        return GridSearchCV(build_pipeline(name), GRIDS[name], **common)
    return RandomizedSearchCV(
        build_pipeline(name),
        DISTRIBUTIONS[name],
        n_iter=config.SEARCH_N_ITER[name],
        random_state=config.SEARCH_SEED,
        **common,
    )


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Tira o prefixo ``model__`` e converte tipos numpy para JSON."""
    out = {}
    for key, value in params.items():
        if hasattr(value, "item"):
            value = value.item()
        out[key.removeprefix("model__")] = value
    return out


def tuned_model(name: str, params: dict[str, Any]) -> Any:  # noqa: ANN401
    """Estimador do catálogo com os hiperparâmetros informados.

    Args:
        name: Chave em ``models.CATALOG``.
        params: Hiperparâmetros (sem prefixo).

    Returns:
        Estimador não ajustado.
    """
    return CATALOG[name].factory().set_params(**params)


def tune_model(name: str) -> dict[str, Any]:
    """Busca, reavalia nos folds padrão e aplica o critério de adoção.

    Args:
        name: Chave em ``models.CATALOG``.

    Returns:
        Resumo: parâmetros, notas da busca e dos folds padrão, otimismo e decisão.
    """
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    search = make_search(name).fit(X, y)
    params = _clean_params(search.cv_results_["params"][search.best_index_])
    default = cv_scores(build_pipeline(name), X, y)
    tuned = cv_scores(build_pipeline(name, tuned_model(name, params)), X, y)
    pr = paired_comparison(default, tuned, "pr_auc")
    brier = paired_comparison(default, tuned, "brier")
    adopt = (
        pr["ganho_medio"] > 0 and pr["win_rate"] > config.WIN_RATE_MIN and brier["win_rate"] >= 0.5
    )
    logger.info("Tuning: %s concluído (adotar=%s)", name, adopt)
    return {
        "modelo": name,
        "candidatos": len(search.cv_results_["params"]),
        "pr_auc_busca": float(search.best_score_),
        "pr_auc_padrao": pr["base"],
        "pr_auc_ajustado": pr["candidato"],
        "otimismo": float(search.best_score_) - pr["candidato"],
        "ganho_pr_auc": pr["ganho_medio"],
        "win_rate_pr_auc": pr["win_rate"],
        "ganho_brier": brier["ganho_medio"],
        "win_rate_brier": brier["win_rate"],
        "adotar": bool(adopt),
        "params": params,
    }


def run_tuning(names: list[str] = TUNABLE) -> pd.DataFrame:
    """Ajusta todos os modelos e salva resultados e parâmetros adotados.

    Args:
        names: Modelos a ajustar.

    Returns:
        Tabela de resultados (``reports/tuning_results.csv``).
    """
    results = [tune_model(n) for n in names]
    adopted = {r["modelo"]: r["params"] for r in results if r["adotar"]}
    config.BEST_PARAMS.write_text(json.dumps(adopted, indent=2, ensure_ascii=False) + "\n")
    table = pd.DataFrame(results)
    table["params"] = table["params"].map(lambda p: json.dumps(p, ensure_ascii=False))
    table.to_csv(config.REPORTS_DIR / "tuning_results.csv", index=False)
    return table


def tuned_leader_duel(leader: str = "xgboost", rival: str = "random_forest") -> pd.DataFrame:
    """Compara os dois melhores modelos já ajustados, nos folds padrão.

    Args:
        leader: Modelo de referência.
        rival: Modelo desafiante.

    Returns:
        Comparação pareada por métrica (salva em ``reports/tuned_duel.csv``).
    """
    df = load_train()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    a = cv_scores(build_pipeline(leader, tuned=True), X, y)
    b = cv_scores(build_pipeline(rival, tuned=True), X, y)
    rows = [
        {"lider": leader, "rival": rival, "métrica": m, **paired_comparison(a, b, m)}
        for m in ["pr_auc", "roc_auc", "ks", "brier"]
    ]
    out = pd.DataFrame(rows)
    out.to_csv(config.REPORTS_DIR / "tuned_duel.csv", index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print(run_tuning().drop(columns="params").round(4).to_string(index=False))
    print(tuned_leader_duel().round(4).to_string(index=False))

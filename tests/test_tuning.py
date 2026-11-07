import json

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from credit_risk import config
from credit_risk.tuning import (
    TUNABLE,
    _clean_params,
    make_search,
    tuned_model,
)


def test_grid_for_simple_models_and_random_for_ensembles():
    assert isinstance(make_search("logistic"), GridSearchCV)
    assert isinstance(make_search("decision_tree"), GridSearchCV)
    assert isinstance(make_search("xgboost"), RandomizedSearchCV)


def test_search_uses_separate_seed_and_pr_auc():
    for name in TUNABLE:
        search = make_search(name)
        assert search.cv.random_state == config.SEARCH_SEED != config.SEED
        assert search.scoring == "average_precision"


def test_clean_params_strips_prefix_and_numpy_types():
    out = _clean_params({"model__max_depth": np.int64(4), "model__learning_rate": np.float64(0.1)})
    assert out == {"max_depth": 4, "learning_rate": 0.1}
    assert json.dumps(out)


def test_tuned_model_applies_params():
    model = tuned_model("decision_tree", {"max_depth": 3})
    assert model.get_params()["max_depth"] == 3


def test_build_pipeline_tuned_reads_best_params(tmp_path, monkeypatch):
    from credit_risk.models import build_pipeline

    path = tmp_path / "best.json"
    path.write_text(json.dumps({"decision_tree": {"max_depth": 3}}))
    monkeypatch.setattr(config, "BEST_PARAMS", path)
    assert build_pipeline("decision_tree", tuned=True).named_steps["model"].max_depth == 3
    assert build_pipeline("decision_tree").named_steps["model"].max_depth == 6
    assert build_pipeline("logistic", tuned=True).named_steps["model"].C == 1.0


def test_build_pipeline_tuned_without_file_uses_defaults(tmp_path, monkeypatch):
    from credit_risk.models import build_pipeline

    monkeypatch.setattr(config, "BEST_PARAMS", tmp_path / "nao_existe.json")
    assert build_pipeline("decision_tree", tuned=True).named_steps["model"].max_depth == 6

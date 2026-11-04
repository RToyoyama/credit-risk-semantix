import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from credit_risk import config
from credit_risk.experiments import decide, model_pipeline


def test_decide_requires_pr_auc_gain_and_win_rate():
    table = pd.DataFrame(
        {
            "métrica": ["pr_auc", "pr_auc", "pr_auc", "roc_auc"],
            "ganho_medio": [0.01, 0.01, -0.01, 0.05],
            "win_rate": [0.80, 0.60, 0.90, 1.0],
        }
    )
    assert decide(table)["adotar"].tolist() == [True, False, False, False]


@pytest.mark.parametrize("engineered", [False, True])
def test_model_pipeline_columns(raw_df, engineered):
    from credit_risk.cleaning import clean

    df = clean(pd.concat([raw_df] * 60, ignore_index=True))
    pipe = model_pipeline(DummyClassifier(), engineered=engineered).fit(df, df[config.TARGET])
    names = pipe.named_steps["prep"].named_steps["prep"].get_feature_names_out()
    assert ("util_mean" in names) is engineered
    assert "limit_bal" in names


def test_decide_with_calibration_vetoes_worse_brier():
    from credit_risk.experiments import decide_with_calibration

    rows = []
    for strat, pr_win, brier_win in [("a", 0.8, 0.6), ("b", 0.8, 0.2), ("c", 0.4, 0.9)]:
        rows.append(
            {
                "modelo": "m",
                "estrategia": strat,
                "métrica": "pr_auc",
                "ganho_medio": 0.01,
                "win_rate": pr_win,
            }
        )
        rows.append(
            {
                "modelo": "m",
                "estrategia": strat,
                "métrica": "brier",
                "ganho_medio": 0.0,
                "win_rate": brier_win,
            }
        )
    out = decide_with_calibration(pd.DataFrame(rows)).set_index("estrategia")
    assert out["adotar"].to_dict() == {"a": True, "b": False, "c": False}

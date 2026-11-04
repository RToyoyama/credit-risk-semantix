import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from credit_risk import config
from credit_risk.experiments import decide, feature_set_pipeline


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
def test_feature_set_pipeline_columns(raw_df, engineered):
    from credit_risk.cleaning import clean

    df = clean(raw_df)
    pipe = feature_set_pipeline(DummyClassifier(), engineered).fit(df, df[config.TARGET])
    n_num = len(config.RAW_NUMERIC_COLS) + (len(config.ENGINEERED_FEATURES) if engineered else 0)
    assert (
        pipe.named_steps["prep"].transformers_[0][2] == pipe.named_steps["prep"].transformers[0][2]
    )
    assert len(pipe.named_steps["prep"].transformers_[0][2]) == n_num

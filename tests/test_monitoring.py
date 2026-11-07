import numpy as np
import pandas as pd
import pytest

from credit_risk import config
from credit_risk.monitoring import _proportions, fit_reference, psi, status


def test_psi_worked_example():
    assert psi([0.5, 0.5], [0.7, 0.3]) == pytest.approx(0.17, abs=0.001)


def test_psi_zero_for_identical_and_handles_empty_bins():
    assert psi([0.2, 0.8], [0.2, 0.8]) == pytest.approx(0.0)
    assert np.isfinite(psi([0.5, 0.5, 0.0], [0.4, 0.4, 0.2]))


@pytest.mark.parametrize(
    ("value", "label"), [(0.05, "estável"), (0.17, "atenção"), (0.4, "investigar")]
)
def test_status_thresholds(value, label):
    assert status(value) == label


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 5_000
    data = {c: rng.normal(size=n) for c in config.MONITORED_NUMERIC}
    data["pay_status_1"] = rng.integers(-2, 4, n)
    data["delay_months"] = rng.integers(0, 7, n)
    return pd.DataFrame(data)


def test_reference_deciles_and_same_population_is_stable(frame):
    ref = fit_reference(frame)
    assert len(ref["score"]["props"]) == config.PSI_BINS
    assert np.allclose(ref["score"]["props"], 0.1, atol=0.01)
    same = frame.sample(frac=0.5, random_state=1)
    assert psi(ref["score"]["props"], _proportions(same["score"], ref["score"])) < 0.02


def test_shifted_population_raises_alarm(frame):
    ref = fit_reference(frame)
    shifted = frame.assign(score=frame["score"] + 1.0)
    assert psi(ref["score"]["props"], _proportions(shifted["score"], ref["score"])) > 0.25


def test_categories_beyond_reference_go_to_extremes(frame):
    ref = fit_reference(frame)
    out = _proportions(pd.Series([10, 10, -5]), ref["pay_status_1"])
    assert out[-1] == pytest.approx(2 / 3) and out[0] == pytest.approx(1 / 3)


def test_run_monitoring_end_to_end_stable_then_alarm(raw_df):
    from credit_risk.cleaning import clean
    from credit_risk.models import final_pipeline
    from credit_risk.monitoring import monitoring_frame, run_monitoring
    from credit_risk.scoring import assign_band, load_params

    df = clean(pd.concat([raw_df] * 80, ignore_index=True))
    rng = np.random.default_rng(0)
    df["pay_status_1"] = rng.integers(-2, 4, len(df))
    df["limit_bal"] = rng.integers(10, 500, len(df)) * 1_000
    df[config.TARGET] = (df["pay_status_1"] >= 2).astype(int)
    X = df.drop(columns=[config.TARGET])
    model = final_pipeline().fit(X, df[config.TARGET])
    frame = monitoring_frame(model, X)
    reference = fit_reference(frame)
    reference["_indicadores"] = {
        "pd_media": float(frame["pd"].mean()),
        "fracao_faixa_e": max(
            float((assign_band(frame["score"], load_params().band_cuts) == "E").mean()), 1e-3
        ),
    }
    same = run_monitoring(X, model, reference)
    assert same["variaveis"]["limit_bal"]["status"] == "estável"
    assert same["indicadores"]["pd_media"]["razao"] == pytest.approx(1.0, abs=0.01)
    risky = run_monitoring(X[X["pay_status_1"] >= 2], model, reference)
    assert risky["status_geral"] == "investigar"

import sys
from pathlib import Path

import pytest

from credit_risk import config

APP = Path(__file__).resolve().parents[1] / "app"
needs_data = pytest.mark.skipif(
    not (config.TRAIN_PARQUET.exists() and config.FINAL_MODEL_PATH.exists()),
    reason="dados e modelo locais ausentes (rode make data split final)",
)


@pytest.fixture(autouse=True)
def _app_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))


def test_money_and_markdown_helpers():
    sys.path.insert(0, str(APP))
    from theme import md, money

    assert money(20, "auto") == "NT$ 20"
    assert money(48_500, "auto") == "NT$ 48,5 mil"
    assert money(7_500_000, "auto") == "NT$ 7,5 mi"
    assert md("NT$ 1 a NT$ 2") == "NT\\$ 1 a NT\\$ 2"


@needs_data
@pytest.mark.parametrize(
    "page",
    [
        "page_problem.py",
        "page_data.py",
        "page_segments.py",
        "page_decision.py",
        "page_portfolio.py",
        "page_monitoring.py",
        "page_simulator.py",
    ],
)
def test_pages_run_without_exceptions(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / page), default_timeout=180).run()
    assert not at.exception
    assert at.title


@needs_data
def test_problem_page_breakeven_matches_decision_module():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "page_problem.py"), default_timeout=180).run()
    text = " ".join(m.value for m in at.markdown)
    assert "**19%**" in text


@needs_data
def test_decision_page_defaults_reproduce_holdout_result():
    import pandas as pd
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "page_decision.py"), default_timeout=180).run()
    policy = pd.read_csv(config.REPORTS_DIR / "holdout_policy.csv").set_index("política")
    expected = policy.loc["maior valor esperado (modelo)", "valor"] / 1e6
    shown = next(m.value for m in at.metric if m.label == "Perda evitada pelo modelo")
    assert shown == f"NT$ {expected:.1f} mi".replace(".", ",")

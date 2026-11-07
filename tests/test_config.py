import importlib
from pathlib import Path

from credit_risk import config


def test_root_is_project_folder_in_editable_install():
    assert (config.ROOT / "pyproject.toml").exists()
    assert config.REPORTS_DIR == config.ROOT / "reports"


def test_root_env_variable_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDIT_RISK_ROOT", str(tmp_path))
    try:
        assert importlib.reload(config).ROOT == tmp_path.resolve()
    finally:
        monkeypatch.delenv("CREDIT_RISK_ROOT")
        importlib.reload(config)


def test_root_falls_back_to_cwd_when_installed_as_copy(tmp_path, monkeypatch):
    fake = tmp_path / "site-packages" / "credit_risk" / "config.py"
    monkeypatch.delenv("CREDIT_RISK_ROOT", raising=False)
    monkeypatch.setattr(config, "__file__", str(fake))
    monkeypatch.chdir(tmp_path)
    assert config._find_root() == Path(tmp_path).resolve()

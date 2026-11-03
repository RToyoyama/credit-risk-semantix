import hashlib

import pandas as pd
import pytest

from credit_risk import config
from credit_risk.data import (
    SchemaError,
    download_raw,
    ensure_valid,
    sha256sum,
    validate_raw,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Tabela sintética pequena, válida segundo o schema da UCI."""
    n = 4
    data = {
        "ID": [1, 2, 3, 4],
        "LIMIT_BAL": [20_000, 50_000, 120_000, 500_000],
        "SEX": [1, 2, 2, 1],
        "EDUCATION": [1, 2, 3, 4],
        "MARRIAGE": [1, 2, 3, 1],
        "AGE": [21, 35, 47, 79],
        **{c: [-2, -1, 0, 2] for c in config.PAY_STATUS_COLS},
        **{c: [-500, 0, 3_000, 90_000] for c in config.BILL_AMT_COLS},
        **{c: [0, 100, 2_000, 5_000] for c in config.PAY_AMT_COLS},
        config.RAW_TARGET: [0, 0, 1, 1],
    }
    assert all(len(v) == n for v in data.values())
    return pd.DataFrame(data)[config.RAW_COLUMNS]


def test_valid_data_has_no_errors_or_warnings(raw_df):
    report = validate_raw(raw_df)
    assert report.ok
    assert report.warnings == []


def test_missing_column_is_error(raw_df):
    report = validate_raw(raw_df.drop(columns="AGE"))
    assert not report.ok
    assert "AGE" in report.errors[0]


def test_extra_column_is_error(raw_df):
    assert not validate_raw(raw_df.assign(EXTRA=1)).ok


def test_duplicated_id_is_error(raw_df):
    raw_df.loc[1, "ID"] = 1
    assert any("duplicados" in e for e in validate_raw(raw_df).errors)


def test_null_is_error(raw_df):
    raw_df["AGE"] = raw_df["AGE"].astype("float").where(raw_df.index != 0)
    errors = validate_raw(raw_df).errors
    assert any("nulos" in e for e in errors)
    assert any("não inteiras" in e for e in errors)


def test_known_undocumented_category_is_warning_not_error(raw_df):
    raw_df.loc[0, "EDUCATION"] = 5
    raw_df.loc[1, "MARRIAGE"] = 0
    report = validate_raw(raw_df)
    assert report.ok
    assert len(report.warnings) == 2


def test_unknown_category_is_error(raw_df):
    raw_df.loc[0, "SEX"] = 3
    assert not validate_raw(raw_df).ok


def test_target_outside_binary_is_error(raw_df):
    raw_df.loc[0, config.RAW_TARGET] = 2
    assert not validate_raw(raw_df).ok


@pytest.mark.parametrize(
    ("col", "value"),
    [("AGE", 17), ("AGE", 101), ("LIMIT_BAL", 0), ("PAY_0", -3), ("PAY_6", 10), ("PAY_AMT1", -1)],
)
def test_out_of_range_is_error(raw_df, col, value):
    raw_df.loc[0, col] = value
    assert not validate_raw(raw_df).ok


def test_negative_bill_amount_is_allowed(raw_df):
    raw_df.loc[0, "BILL_AMT1"] = -165_580
    assert validate_raw(raw_df).ok


def test_ensure_valid_raises_on_error(raw_df):
    with pytest.raises(SchemaError):
        ensure_valid(raw_df.drop(columns="SEX"))


@pytest.fixture
def source_file(tmp_path):
    src = tmp_path / "source.zip"
    src.write_bytes(b"conteudo de teste")
    return src, hashlib.sha256(b"conteudo de teste").hexdigest()


def test_download_checks_hash(tmp_path, source_file):
    src, digest = source_file
    dest = download_raw(url=src.as_uri(), dest=tmp_path / "out.zip", expected_sha256=digest)
    assert sha256sum(dest) == digest


def test_download_rejects_wrong_hash_and_leaves_nothing(tmp_path, source_file):
    src, _ = source_file
    dest = tmp_path / "out.zip"
    with pytest.raises(ValueError, match="SHA-256"):
        download_raw(url=src.as_uri(), dest=dest, expected_sha256="0" * 64)
    assert list(tmp_path.glob("out.*")) == []


def test_download_skips_when_file_is_intact(tmp_path, source_file):
    src, digest = source_file
    dest = tmp_path / "out.zip"
    download_raw(url=src.as_uri(), dest=dest, expected_sha256=digest)
    src.unlink()
    assert download_raw(url=src.as_uri(), dest=dest, expected_sha256=digest) == dest

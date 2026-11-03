from credit_risk import config
from credit_risk.cleaning import clean, count_status_bill_mismatch, rename_columns


def test_rename_produces_clean_schema(raw_df):
    out = rename_columns(raw_df)
    assert list(out.columns) == list(config.RENAME_MAP.values())
    assert "pay_status_1" in out and "PAY_0" not in out


def test_rename_aligns_months(raw_df):
    raw_df.loc[0, "PAY_0"] = 7
    raw_df.loc[0, "BILL_AMT1"] = 12_345
    out = rename_columns(raw_df)
    assert out.loc[0, "pay_status_1"] == 7
    assert out.loc[0, "bill_amt_1"] == 12_345


def test_undocumented_categories_are_grouped_and_flagged(raw_df):
    raw_df["EDUCATION"] = [0, 5, 6, 1]
    raw_df["MARRIAGE"] = [0, 1, 2, 3]
    out = clean(raw_df)
    assert out["education"].tolist() == [4, 4, 4, 1]
    assert out["education_undocumented"].tolist() == [1, 1, 1, 0]
    assert out["marriage"].tolist() == [3, 1, 2, 3]
    assert out["marriage_undocumented"].tolist() == [1, 0, 0, 0]


def test_clean_keeps_rows_ids_and_target(raw_df):
    out = clean(raw_df)
    assert len(out) == len(raw_df)
    assert out["id"].tolist() == raw_df["ID"].tolist()
    assert out[config.TARGET].tolist() == raw_df[config.RAW_TARGET].tolist()


def test_clean_does_not_modify_input(raw_df):
    raw_df["EDUCATION"] = [0, 5, 6, 1]
    before = raw_df.copy()
    clean(raw_df)
    assert raw_df.equals(before)


def test_status_bill_mismatch_counts_both_contradictions(raw_df):
    df = rename_columns(raw_df)
    df[config.PAY_STATUS] = -1
    df[config.BILL_AMT] = 1_000
    df.loc[0, "pay_status_1"], df.loc[0, "bill_amt_1"] = -2, 500  # sem consumo, fatura > 0
    df.loc[0, "pay_status_2"], df.loc[0, "bill_amt_2"] = 2, 0  # atraso, fatura = 0
    df.loc[1, "pay_status_3"], df.loc[1, "bill_amt_3"] = 1, -10  # atraso, fatura < 0
    assert count_status_bill_mismatch(df).tolist() == [2, 1, 0, 0]

import pandas as pd
import pytest

from credit_risk import config
from credit_risk.split import get_holdout_ids, make_holdout_ids, split_by_ids


def test_holdout_has_expected_size(clean_df):
    assert len(make_holdout_ids(clean_df)) == 200


def test_train_and_holdout_do_not_overlap_and_cover_all(clean_df):
    train, holdout = split_by_ids(clean_df, make_holdout_ids(clean_df))
    assert set(train["id"]).isdisjoint(holdout["id"])
    assert set(train["id"]) | set(holdout["id"]) == set(clean_df["id"])
    assert len(train) + len(holdout) == len(clean_df)


def test_split_is_stratified(clean_df):
    train, holdout = split_by_ids(clean_df, make_holdout_ids(clean_df))
    assert abs(train[config.TARGET].mean() - holdout[config.TARGET].mean()) < 0.01


def test_split_is_reproducible_and_seed_dependent(clean_df):
    a = make_holdout_ids(clean_df, seed=42)
    assert a.equals(make_holdout_ids(clean_df, seed=42))
    assert not a.equals(make_holdout_ids(clean_df, seed=7))


def test_sealed_ids_file_wins_over_resampling(tmp_path, clean_df):
    path = tmp_path / "holdout_ids.csv"
    sealed = pd.Series([1, 2, 3], name="id")
    sealed.to_frame().to_csv(path, index=False)
    assert get_holdout_ids(clean_df, path).tolist() == [1, 2, 3]


def test_ids_file_is_created_when_absent(tmp_path, clean_df):
    path = tmp_path / "holdout_ids.csv"
    ids = get_holdout_ids(clean_df, path)
    assert path.exists()
    assert pd.read_csv(path)["id"].equals(ids)


def test_unknown_holdout_id_raises(clean_df):
    with pytest.raises(ValueError, match="não existem"):
        split_by_ids(clean_df, pd.Series([999_999]))

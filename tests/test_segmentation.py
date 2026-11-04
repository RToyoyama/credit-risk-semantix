import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from credit_risk import config
from credit_risk.segmentation import (
    SegmentEncoder,
    choose_k,
    choose_n_components,
    explained_variance_table,
    kmeans_selection,
    loadings,
    make_pca_pipeline,
    segment_profile,
    stability,
)


@pytest.fixture
def clean_sample() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    level = rng.random(n)
    df = pd.DataFrame({"limit_bal": rng.integers(10, 100, n) * 1_000})
    for i, (b, p, s) in enumerate(
        zip(config.BILL_AMT, config.PAY_AMT, config.PAY_STATUS, strict=True)
    ):
        df[b] = (level * df["limit_bal"] * (1 + 0.05 * i)).round()
        df[p] = (rng.random(n) * 3_000).round()
        df[s] = rng.integers(-2, 3, n)
    return df


@pytest.mark.parametrize(
    ("cumulative", "expected"),
    [([0.5, 0.79, 0.85, 1.0], 3), ([0.8, 0.9, 1.0], 1), ([0.2, 0.5, 1.0], 3)],
)
def test_choose_n_components(cumulative, expected):
    assert choose_n_components(pd.Series(cumulative)) == expected


def test_pca_pipeline_variance_table(clean_sample):
    pipe = make_pca_pipeline().fit(clean_sample)
    table = explained_variance_table(pipe.named_steps["pca"])
    assert len(table) == len(config.BEHAVIOR_FEATURES)
    assert table["acumulada"].iloc[-1] == pytest.approx(1.0)
    assert table["variancia"].is_monotonic_decreasing


def test_loadings_are_unit_vectors(clean_sample):
    pipe = make_pca_pipeline(3).fit(clean_sample)
    load = loadings(pipe.named_steps["pca"], config.BEHAVIOR_FEATURES)
    assert load.shape == (17, 3)
    assert np.allclose((load**2).sum(axis=0), 1.0)


def test_pca_pipeline_is_clonable_and_reproducible(clean_sample):
    a = make_pca_pipeline(3).fit_transform(clean_sample)
    b = clone(make_pca_pipeline(3)).fit_transform(clean_sample)
    assert np.allclose(a, b)


def test_segment_encoder_relabels_chronic_delay_last(clean_sample):
    late = clean_sample.index < 40
    clean_sample.loc[late, config.PAY_STATUS] = 4
    clean_sample.loc[~late, config.PAY_STATUS] = 0
    enc = SegmentEncoder(n_segments=3).fit(clean_sample)
    seg = enc.predict(clean_sample)
    assert set(seg) <= {0, 1, 2}
    assert (seg[late] == 2).mean() > 0.9


def test_segment_encoder_does_not_use_target(clean_sample):
    a = SegmentEncoder().fit(clean_sample).predict(clean_sample)
    b = SegmentEncoder().fit(clean_sample, y=np.arange(len(clean_sample))).predict(clean_sample)
    assert (a == b).all()


def test_segment_encoder_one_hot_and_clone(clean_sample):
    enc = clone(SegmentEncoder(one_hot=True)).fit(clean_sample)
    out = enc.transform(clean_sample)
    assert out.shape == (len(clean_sample), config.N_SEGMENTS)
    assert (out.sum(axis=1) == 1).all()
    assert list(enc.get_feature_names_out()) == ["segment_0", "segment_1", "segment_2"]


def test_choose_k_respects_min_share():
    table = pd.DataFrame(
        {"k": [3, 4, 5], "silhueta": [0.4, 0.6, 0.5], "menor_grupo": [0.10, 0.01, 0.08]}
    )
    assert choose_k(table) == 5


def test_choose_k_raises_when_no_k_is_valid():
    table = pd.DataFrame({"k": [3], "silhueta": [0.4], "menor_grupo": [0.01]})
    with pytest.raises(ValueError):
        choose_k(table)


def test_kmeans_selection_and_stability_on_separated_blobs():
    rng = np.random.default_rng(1)
    scores = np.vstack([rng.normal(loc, 0.1, size=(100, 2)) for loc in (0, 5, 10)])
    table = kmeans_selection(scores, ks=range(2, 5))
    assert choose_k(table) == 3
    stab = stability(scores, 3, n_seeds=3, n_subsamples=3)
    assert stab["ari_min_sementes"] == pytest.approx(1.0)
    assert stab["ari_min_subamostras"] == pytest.approx(1.0)


def test_segment_profile_has_one_row_per_segment(clean_sample):
    seg = np.arange(len(clean_sample)) % 3
    prof = segment_profile(clean_sample, seg)
    assert list(prof.index) == [0, 1, 2]
    assert prof["clientes"].sum() == len(clean_sample)
    assert prof["participacao"].sum() == pytest.approx(1.0)

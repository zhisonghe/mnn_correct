import numpy as np
import anndata as ad
import pandas as pd
import pytest

import mnn_correct
from mnn_correct import MNNCorrector, mnn_correct as mnn_correct_fn, mnn_correct_adata


# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures
# ──────────────────────────────────────────────────────────────────────────── #

def _make_batch(n: int = 40, d: int = 8, shift: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) + shift


def _make_adata_pair(
    n_ref: int = 40, n_query: int = 40, d: int = 8, shift: float = 2.0, seed: int = 0
):
    """Returns (adata_ref, adata_query) with a known shift between batches."""
    emb_ref = _make_batch(n_ref, d, 0.0, seed)
    emb_query = _make_batch(n_query, d, shift, seed + 1)

    def _adata(emb, prefix):
        obs = pd.DataFrame(index=[f"{prefix}_{i}" for i in range(len(emb))])
        a = ad.AnnData(X=np.zeros((len(emb), 1)), obs=obs)
        a.obsm["X_test"] = emb
        return a

    return _adata(emb_ref, "ref"), _adata(emb_query, "query")


def _make_combined_adata(
    n_per_batch: int = 40, d: int = 8, n_batches: int = 3, seed: int = 0
) -> ad.AnnData:
    """Returns a combined AnnData with n_batches batches, each shifted."""
    rng = np.random.default_rng(seed)
    adatas = []
    for b in range(n_batches):
        emb = rng.standard_normal((n_per_batch, d)) + b * 2.0
        obs = pd.DataFrame(
            {"batch": [f"batch{b}"] * n_per_batch},
            index=[f"b{b}_cell{i}" for i in range(n_per_batch)],
        )
        a = ad.AnnData(X=np.zeros((n_per_batch, 1)), obs=obs)
        a.obsm["X_test"] = emb
        adatas.append(a)
    return ad.concat(adatas)


# ──────────────────────────────────────────────────────────────────────────── #
# Package-level smoke tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_version() -> None:
    assert mnn_correct.__version__ == "0.1.0"


def test_public_api() -> None:
    assert hasattr(mnn_correct, "MNNCorrector")
    assert hasattr(mnn_correct, "mnn_correct")
    assert hasattr(mnn_correct, "mnn_correct_adata")


# ──────────────────────────────────────────────────────────────────────────── #
# MNNCorrector unit tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_corrector_fit_stores_state() -> None:
    emb_ref = _make_batch(40, 8, 0.0)
    emb_query = _make_batch(40, 8, 2.0, seed=1)
    corrector = MNNCorrector(k_mnn=5, k_propagate=10, verbose=False)
    corrector.fit(emb_ref, emb_query)

    assert corrector.is_fitted
    assert corrector.n_mnn_pairs_ > 0
    assert corrector.n_query_with_mnn_ > 0
    assert corrector.emb_query_with_mnn_ is not None
    assert corrector.avg_disp_ is not None
    assert corrector.avg_disp_.shape[1] == 8


def test_corrector_transform_shape() -> None:
    emb_ref = _make_batch(40, 8, 0.0)
    emb_query = _make_batch(40, 8, 2.0, seed=1)
    corrector = MNNCorrector(k_mnn=5, k_propagate=10, verbose=False)
    corrector.fit(emb_ref, emb_query)

    emb_new = _make_batch(20, 8, 2.0, seed=99)
    corrected = corrector.transform(emb_new)
    assert corrected.shape == emb_new.shape


def test_corrector_transform_reduces_distance() -> None:
    """Transformed query cells should be closer to ref centroid."""
    emb_ref = _make_batch(60, 8, 0.0)
    emb_query = _make_batch(60, 8, 3.0, seed=1)
    corrector = MNNCorrector(k_mnn=5, k_propagate=15, verbose=False)
    corrector.fit(emb_ref, emb_query)

    emb_new = _make_batch(30, 8, 3.0, seed=42)
    corrected = corrector.transform(emb_new)

    ref_center = emb_ref.mean(axis=0)
    assert np.linalg.norm(corrected.mean(axis=0) - ref_center) < \
           np.linalg.norm(emb_new.mean(axis=0) - ref_center)


def test_corrector_not_fitted_transform_raises() -> None:
    corrector = MNNCorrector(verbose=False)
    with pytest.raises(RuntimeError, match="fitted"):
        corrector.transform(np.zeros((5, 8)))


def test_corrector_fit_transform_equivalent() -> None:
    emb_ref = _make_batch(40, 8, 0.0)
    emb_query = _make_batch(40, 8, 2.0, seed=1)

    c1 = MNNCorrector(k_mnn=5, k_propagate=10, verbose=False)
    out1 = c1.fit_transform(emb_ref, emb_query)

    c2 = MNNCorrector(k_mnn=5, k_propagate=10, verbose=False)
    c2.fit(emb_ref, emb_query)
    out2 = c2.transform(emb_query)

    np.testing.assert_array_almost_equal(out1, out2)


def test_corrector_obs_names_indexed() -> None:
    emb_ref = _make_batch(40, 8)
    emb_query = _make_batch(40, 8, 2.0, seed=1)
    names = pd.Index([f"cell_{i}" for i in range(40)])
    corrector = MNNCorrector(k_mnn=5, verbose=False)
    corrector.fit(emb_ref, emb_query, obs_names_query=names)

    assert corrector.obs_names_with_mnn_ is not None
    assert corrector.obs_names_with_mnn_.isin(names).all()


# ──────────────────────────────────────────────────────────────────────────── #
# mnn_correct (pairwise) tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_mnn_correct_returns_corrector() -> None:
    adata_ref, adata_query = _make_adata_pair()
    result = mnn_correct_fn(adata_ref, adata_query, use_rep="X_test",
                            k_mnn=5, k_propagate=10, verbose=False)
    assert isinstance(result, MNNCorrector)
    assert result.is_fitted


def test_mnn_correct_writes_obsm() -> None:
    adata_ref, adata_query = _make_adata_pair()
    mnn_correct_fn(adata_ref, adata_query, use_rep="X_test",
                   k_mnn=5, k_propagate=10, verbose=False)
    assert "X_test_mnn_corrected" in adata_query.obsm


def test_mnn_correct_key_added() -> None:
    adata_ref, adata_query = _make_adata_pair()
    mnn_correct_fn(adata_ref, adata_query, use_rep="X_test",
                   key_added="X_my_corrected", k_mnn=5, k_propagate=10, verbose=False)
    assert "X_my_corrected" in adata_query.obsm


def test_mnn_correct_ref_unchanged() -> None:
    adata_ref, adata_query = _make_adata_pair()
    emb_ref_before = adata_ref.obsm["X_test"].copy()
    mnn_correct_fn(adata_ref, adata_query, use_rep="X_test",
                   k_mnn=5, k_propagate=10, verbose=False)
    np.testing.assert_array_equal(adata_ref.obsm["X_test"], emb_ref_before)


def test_mnn_correct_query_shift_reduced() -> None:
    adata_ref, adata_query = _make_adata_pair(shift=3.0)
    mnn_correct_fn(adata_ref, adata_query, use_rep="X_test",
                   k_mnn=5, k_propagate=10, verbose=False)

    ref_center = adata_ref.obsm["X_test"].mean(axis=0)
    query_before = adata_query.obsm["X_test"].mean(axis=0)
    query_after = adata_query.obsm["X_test_mnn_corrected"].mean(axis=0)

    assert np.linalg.norm(query_after - ref_center) < \
           np.linalg.norm(query_before - ref_center)


def test_mnn_correct_missing_use_rep_raises() -> None:
    adata_ref, adata_query = _make_adata_pair()
    with pytest.raises(KeyError, match="use_rep"):
        mnn_correct_fn(adata_ref, adata_query, use_rep="X_missing", verbose=False)


# ──────────────────────────────────────────────────────────────────────────── #
# mnn_correct_adata tests
# ──────────────────────────────────────────────────────────────────────────── #

def test_mnn_correct_adata_sequential_two_batches() -> None:
    adata = _make_combined_adata(n_per_batch=40, n_batches=2)
    _, correctors = mnn_correct_adata(
        adata, batch_key="batch", use_rep="X_test",
        batch_order=["batch0", "batch1"],
        k_mnn=5, k_propagate=10, verbose=False,
    )
    assert "X_test_mnn_corrected" in adata.obsm
    assert len(correctors) == 1


def test_mnn_correct_adata_sequential_three_batches() -> None:
    adata = _make_combined_adata(n_per_batch=40, n_batches=3)
    _, correctors = mnn_correct_adata(
        adata, batch_key="batch", use_rep="X_test",
        batch_order=["batch0", "batch1", "batch2"],
        k_mnn=5, k_propagate=10, verbose=False,
    )
    assert "X_test_mnn_corrected" in adata.obsm
    assert len(correctors) == 2  # 2 correction rounds


def test_mnn_correct_adata_ref_batch_unchanged() -> None:
    adata = _make_combined_adata(n_per_batch=40, n_batches=3)
    ref_emb_before = adata[adata.obs["batch"] == "batch0"].obsm["X_test"].copy()

    mnn_correct_adata(
        adata, batch_key="batch", use_rep="X_test",
        batch_order=["batch0", "batch1", "batch2"],
        k_mnn=5, k_propagate=10, verbose=False,
    )

    ref_emb_after = adata[adata.obs["batch"] == "batch0"].obsm["X_test_mnn_corrected"]
    np.testing.assert_array_equal(ref_emb_after, ref_emb_before)


def test_mnn_correct_adata_fixed_reference() -> None:
    adata = _make_combined_adata(n_per_batch=40, n_batches=3)
    _, correctors = mnn_correct_adata(
        adata, batch_key="batch", use_rep="X_test",
        reference="batch0",
        k_mnn=5, k_propagate=10, verbose=False,
    )
    assert "X_test_mnn_corrected" in adata.obsm
    assert len(correctors) == 2  # 2 non-reference batches


def test_mnn_correct_adata_copy() -> None:
    adata = _make_combined_adata(n_per_batch=40, n_batches=2)
    adata_out, _ = mnn_correct_adata(
        adata, batch_key="batch", use_rep="X_test",
        batch_order=["batch0", "batch1"],
        k_mnn=5, k_propagate=10, copy=True, verbose=False,
    )
    assert adata_out is not adata
    assert "X_test_mnn_corrected" not in adata.obsm  # original untouched
    assert "X_test_mnn_corrected" in adata_out.obsm


def test_mnn_correct_adata_alphabetical_fallback() -> None:
    """Without batch_order or reference, batches should be sorted alphabetically."""
    adata = _make_combined_adata(n_per_batch=40, n_batches=2)
    _, correctors = mnn_correct_adata(
        adata, batch_key="batch", use_rep="X_test",
        k_mnn=5, k_propagate=10, verbose=False,
    )
    assert len(correctors) == 1


def test_mnn_correct_adata_both_order_and_ref_raises() -> None:
    adata = _make_combined_adata(n_per_batch=40, n_batches=2)
    with pytest.raises(ValueError, match="not both"):
        mnn_correct_adata(
            adata, batch_key="batch", use_rep="X_test",
            reference="batch0", batch_order=["batch0", "batch1"],
            verbose=False,
        )


def test_mnn_correct_adata_missing_batch_key_raises() -> None:
    adata = _make_combined_adata(n_per_batch=40, n_batches=2)
    with pytest.raises(KeyError, match="batch_key"):
        mnn_correct_adata(adata, batch_key="nonexistent", use_rep="X_test", verbose=False)


def test_mnn_correct_adata_missing_reference_raises() -> None:
    adata = _make_combined_adata(n_per_batch=40, n_batches=2)
    with pytest.raises(ValueError, match="reference"):
        mnn_correct_adata(adata, batch_key="batch", use_rep="X_test",
                          reference="ghost_batch", verbose=False)


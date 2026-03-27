import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import mnn_correct
from mnn_correct import MNNCorrector, mnn_correct as mnn_correct_fn, mnn_correct_adata
from mnn_correct.util import wknn


def _make_batch(n: int = 40, d: int = 8, shift: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) + shift


def _make_combined_adata(
    n_per_batch: int = 40,
    d: int = 8,
    n_batches: int = 3,
    shift_scale: float = 2.0,
    seed: int = 0,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    adatas = []
    for batch_index in range(n_batches):
        emb = rng.standard_normal((n_per_batch, d)) + batch_index * shift_scale
        obs = pd.DataFrame(
            {"batch": [f"batch{batch_index}"] * n_per_batch},
            index=[f"b{batch_index}_cell{i}" for i in range(n_per_batch)],
        )
        adata = ad.AnnData(X=np.zeros((n_per_batch, 4)), obs=obs)
        adata.obsm["X_test"] = emb
        adatas.append(adata)
    return ad.concat(adatas)


def _make_adata_pair(
    n_ref: int = 40,
    n_query: int = 40,
    d: int = 8,
    shift: float = 2.0,
    seed: int = 0,
) -> tuple[ad.AnnData, ad.AnnData]:
    emb_ref = _make_batch(n_ref, d, 0.0, seed)
    emb_query = _make_batch(n_query, d, shift, seed + 1)

    ref_obs = pd.DataFrame(index=[f"ref_{i}" for i in range(n_ref)])
    query_obs = pd.DataFrame(index=[f"query_{i}" for i in range(n_query)])

    adata_ref = ad.AnnData(X=np.zeros((n_ref, 4)), obs=ref_obs)
    adata_query = ad.AnnData(X=np.zeros((n_query, 4)), obs=query_obs)
    adata_ref.obsm["X_test"] = emb_ref
    adata_query.obsm["X_test"] = emb_query
    return adata_ref, adata_query


def _make_sparse_adata_pair(
    n_ref: int = 40,
    n_query: int = 40,
    d: int = 12,
    shift: float = 1.0,
    seed: int = 0,
) -> tuple[ad.AnnData, ad.AnnData]:
    rng = np.random.default_rng(seed)
    ref = rng.poisson(1.5, size=(n_ref, d)).astype(float)
    query = rng.poisson(1.5, size=(n_query, d)).astype(float)
    query[:, : max(1, d // 3)] += shift

    adata_ref = ad.AnnData(
        X=sparse.csr_matrix(ref),
        obs=pd.DataFrame(index=[f"ref_sparse_{i}" for i in range(n_ref)]),
    )
    adata_query = ad.AnnData(
        X=sparse.csr_matrix(query),
        obs=pd.DataFrame(index=[f"query_sparse_{i}" for i in range(n_query)]),
    )
    return adata_ref, adata_query


def _make_sparse_combined_adata(
    n_per_batch: int = 40,
    d: int = 12,
    n_batches: int = 2,
    shift_scale: float = 1.0,
    seed: int = 0,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    adatas = []
    feature_span = max(1, d // 3)
    for batch_index in range(n_batches):
        X = rng.poisson(1.5, size=(n_per_batch, d)).astype(float)
        X[:, :feature_span] += batch_index * shift_scale
        obs = pd.DataFrame(
            {"batch": [f"batch{batch_index}"] * n_per_batch},
            index=[f"s{batch_index}_cell{i}" for i in range(n_per_batch)],
        )
        adatas.append(ad.AnnData(X=sparse.csr_matrix(X), obs=obs))
    return ad.concat(adatas)


def test_version() -> None:
    assert mnn_correct.__version__ == "0.1.0"


def test_public_api() -> None:
    assert hasattr(mnn_correct, "MNNCorrector")
    assert hasattr(mnn_correct, "mnn_correct")
    assert hasattr(mnn_correct, "mnn_correct_adata")


def test_build_nn_sklearn_flavor() -> None:
    ref = _make_batch(20, 5, seed=1)
    query = _make_batch(10, 5, seed=2)

    adj = wknn.build_nn(ref=ref, query=query, k=3, flavor="sklearn", verbose=False)

    assert adj.shape == (10, 20)
    assert np.all(np.asarray(adj.sum(axis=1)).ravel() == 3)


def test_build_nn_pynndescent_flavor() -> None:
    ref = _make_batch(30, 5, seed=3)
    query = _make_batch(12, 5, seed=4)

    adj = wknn.build_nn(ref=ref, query=query, k=4, flavor="pynndescent", verbose=False)

    assert adj.shape == (12, 30)
    assert np.all(np.asarray(adj.sum(axis=1)).ravel() == 4)


def test_build_nn_auto_prefers_sklearn_for_small_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = _make_batch(50, 5, seed=5)
    query = _make_batch(40, 5, seed=6)

    def _fail_nn_descent(*args: object, **kwargs: object) -> object:
        raise AssertionError("pynndescent should not be used for small auto inputs")

    monkeypatch.setattr(wknn, "NNDescent", _fail_nn_descent)

    adj = wknn.build_nn(ref=ref, query=query, k=3, flavor="auto", verbose=False)

    assert adj.shape == (40, 50)


def test_build_nn_auto_falls_back_to_pynndescent_for_large_cpu_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _make_batch(1000, 5, seed=7)
    query = _make_batch(1000, 5, seed=8)
    calls: list[str] = []

    class DummyNNDescent:
        def __init__(self, data: np.ndarray) -> None:
            calls.append("init")
            self._data = data

        def query(self, query_data: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            calls.append("query")
            indices = np.tile(np.arange(k), (query_data.shape[0], 1))
            distances = np.zeros((query_data.shape[0], k), dtype=float)
            return indices, distances

    monkeypatch.setattr(wknn, "NNDescent", DummyNNDescent)
    monkeypatch.setattr(wknn.torch.cuda, "is_available", lambda: False)

    adj = wknn.build_nn(ref=ref, query=query, k=3, flavor="auto", verbose=False)

    assert adj.shape == (1000, 1000)
    assert calls == ["init", "query"]


def test_build_mutual_nn_sklearn_flavor() -> None:
    dat1 = _make_batch(18, 5, seed=9)
    dat2 = _make_batch(15, 5, seed=10)

    adj = wknn.build_mutual_nn(dat1=dat1, dat2=dat2, k1=3, flavor="sklearn", verbose=False)

    assert adj.shape == (18, 15)
    assert adj.nnz >= 0


def test_build_mutual_nn_auto_prefers_sklearn_for_small_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dat1 = _make_batch(40, 5, seed=11)
    dat2 = _make_batch(35, 5, seed=12)

    def _fail_nn_descent(*args: object, **kwargs: object) -> object:
        raise AssertionError("pynndescent should not be used for small auto mutual NN inputs")

    monkeypatch.setattr(wknn, "NNDescent", _fail_nn_descent)

    adj = wknn.build_mutual_nn(dat1=dat1, dat2=dat2, k1=3, flavor="auto", verbose=False)

    assert adj.shape == (40, 35)


def test_corrector_fit_stores_batch_models() -> None:
    adata = _make_combined_adata(n_batches=3)
    corrector = MNNCorrector(k_mnn=5, k_propagate=10, verbose=False)

    result = corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1", "batch2"],
        use_rep="X_test",
    )

    assert result is None
    assert corrector.is_fitted
    assert corrector.n_corrections_ == 2
    assert set(corrector.projection_data_) == {"batch1", "batch2"}
    assert corrector.projection_identity_batch_ == "batch0"
    assert corrector.key_added_ == "X_test_mnn_corrected"
    assert corrector.n_mnn_pairs_ > 0
    assert corrector.n_query_with_mnn_ > 0


def test_corrector_fit_can_optionally_return_corrector() -> None:
    adata = _make_combined_adata(n_batches=2)
    corrector = MNNCorrector(k_mnn=5, k_propagate=10, verbose=False)

    result = corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep="X_test",
        return_corrector=True,
    )

    assert result is corrector


def test_corrector_correct_writes_obsm() -> None:
    adata = _make_combined_adata(n_batches=2)
    corrector = MNNCorrector(k_mnn=5, k_propagate=10, verbose=False)
    corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep="X_test",
    )

    corrector.correct(adata)

    assert "X_test_mnn_corrected" in adata.obsm
    assert adata.obsm["X_test_mnn_corrected"].shape == adata.obsm["X_test"].shape


def test_corrector_correct_reduces_distance() -> None:
    adata = _make_combined_adata(n_batches=2, shift_scale=3.0)
    corrector = MNNCorrector(k_mnn=5, k_propagate=10, verbose=False)
    corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep="X_test",
    )
    corrector.correct(adata)

    ref_mask = adata.obs["batch"] == "batch0"
    query_mask = adata.obs["batch"] == "batch1"
    ref_center = adata.obsm["X_test"][ref_mask].mean(axis=0)
    before = adata.obsm["X_test"][query_mask].mean(axis=0)
    after = adata.obsm["X_test_mnn_corrected"][query_mask].mean(axis=0)

    assert np.linalg.norm(after - ref_center) < np.linalg.norm(before - ref_center)


def test_corrector_project_writes_obsm() -> None:
    adata = _make_combined_adata(n_batches=2, shift_scale=3.0)
    corrector = MNNCorrector(k_mnn=5, k_propagate=12, verbose=False)
    corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep="X_test",
    )

    new_obs = pd.DataFrame(index=[f"new_{i}" for i in range(20)])
    new_obs["batch"] = "batch1"
    adata_new = ad.AnnData(X=np.zeros((20, 4)), obs=new_obs)
    adata_new.obsm["X_test"] = _make_batch(20, 8, 3.0, seed=99)

    corrector.project(adata_new, batch_key="batch")

    assert "X_test_mnn_corrected" in adata_new.obsm
    assert adata_new.obsm["X_test_mnn_corrected"].shape == adata_new.obsm["X_test"].shape


def test_corrector_project_reduces_distance() -> None:
    adata = _make_combined_adata(n_batches=2, shift_scale=3.0)
    corrector = MNNCorrector(k_mnn=5, k_propagate=12, verbose=False)
    corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep="X_test",
    )

    adata_new = ad.AnnData(
        X=np.zeros((20, 4)),
        obs=pd.DataFrame({"batch": ["batch1"] * 20}, index=[f"new_{i}" for i in range(20)]),
    )
    adata_new.obsm["X_test"] = _make_batch(20, 8, 3.0, seed=101)
    corrector.project(adata_new, batch_key="batch")

    ref_mask = adata.obs["batch"] == "batch0"
    ref_center = adata.obsm["X_test"][ref_mask].mean(axis=0)
    before = adata_new.obsm["X_test"].mean(axis=0)
    after = adata_new.obsm["X_test_mnn_corrected"].mean(axis=0)

    assert np.linalg.norm(after - ref_center) < np.linalg.norm(before - ref_center)


def test_corrector_project_first_batch_is_identity() -> None:
    adata = _make_combined_adata(n_batches=2, shift_scale=3.0)
    corrector = MNNCorrector(k_mnn=5, k_propagate=12, verbose=False)
    corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep="X_test",
    )

    adata_new = ad.AnnData(
        X=np.zeros((12, 4)),
        obs=pd.DataFrame({"batch": ["batch0"] * 12}, index=[f"new_ref_{i}" for i in range(12)]),
    )
    adata_new.obsm["X_test"] = _make_batch(12, 8, 0.0, seed=202)

    corrector.project(adata_new, batch_key="batch")

    np.testing.assert_allclose(adata_new.obsm["X_test_mnn_corrected"], adata_new.obsm["X_test"])


def test_corrector_project_fixed_reference_is_identity() -> None:
    adata = _make_combined_adata(n_batches=3, shift_scale=3.0)
    corrector = MNNCorrector(k_mnn=5, k_propagate=12, verbose=False)
    corrector.fit(
        adata,
        batch_key="batch",
        reference="batch1",
        use_rep="X_test",
    )

    adata_new = ad.AnnData(
        X=np.zeros((12, 4)),
        obs=pd.DataFrame({"batch": ["batch1"] * 12}, index=[f"new_fixed_ref_{i}" for i in range(12)]),
    )
    adata_new.obsm["X_test"] = _make_batch(12, 8, 3.0, seed=303)

    corrector.project(adata_new, batch_key="batch")

    np.testing.assert_allclose(adata_new.obsm["X_test_mnn_corrected"], adata_new.obsm["X_test"])


def test_corrector_project_mixed_batches_uses_per_batch_models() -> None:
    adata = _make_combined_adata(n_batches=2, shift_scale=3.0)
    corrector = MNNCorrector(k_mnn=5, k_propagate=12, verbose=False)
    corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep="X_test",
    )

    emb_ref = _make_batch(10, 8, 0.0, seed=404)
    emb_query = _make_batch(10, 8, 3.0, seed=405)
    adata_new = ad.AnnData(
        X=np.zeros((20, 4)),
        obs=pd.DataFrame(
            {"batch": ["batch0"] * 10 + ["batch1"] * 10},
            index=[f"mixed_{i}" for i in range(20)],
        ),
    )
    adata_new.obsm["X_test"] = np.vstack([emb_ref, emb_query])

    corrector.project(adata_new, batch_key="batch")

    np.testing.assert_allclose(
        adata_new.obsm["X_test_mnn_corrected"][:10],
        adata_new.obsm["X_test"][:10],
    )
    ref_center = adata.obsm["X_test"][adata.obs["batch"] == "batch0"].mean(axis=0)
    before = adata_new.obsm["X_test"][10:].mean(axis=0)
    after = adata_new.obsm["X_test_mnn_corrected"][10:].mean(axis=0)
    assert np.linalg.norm(after - ref_center) < np.linalg.norm(before - ref_center)


def test_corrector_correct_before_fit_raises() -> None:
    corrector = MNNCorrector(verbose=False)
    adata = _make_combined_adata(n_batches=2)

    with pytest.raises(RuntimeError, match="fit"):
        corrector.correct(adata)


def test_corrector_project_unknown_batch_raises() -> None:
    adata = _make_combined_adata(n_batches=2)
    corrector = MNNCorrector(k_mnn=5, verbose=False)
    corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep="X_test",
    )

    adata_new = ad.AnnData(
        X=np.zeros((5, 4)),
        obs=pd.DataFrame({"batch": ["missing"] * 5}, index=[f"new_{i}" for i in range(5)]),
    )
    adata_new.obsm["X_test"] = _make_batch(5, 8, 2.0, seed=123)

    with pytest.raises(ValueError, match="missing"):
        corrector.project(adata_new, batch_key="batch")


def test_corrector_project_ignores_unused_categorical_levels() -> None:
    adata = _make_combined_adata(n_batches=2, shift_scale=3.0)
    corrector = MNNCorrector(k_mnn=5, k_propagate=12, verbose=False)
    corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep="X_test",
    )

    adata_new = ad.AnnData(
        X=np.zeros((10, 4)),
        obs=pd.DataFrame(
            {
                "batch": pd.Categorical(
                    ["batch1"] * 10,
                    categories=["batch0", "batch1", "unused_batch"],
                )
            },
            index=[f"cat_{i}" for i in range(10)],
        ),
    )
    adata_new.obsm["X_test"] = _make_batch(10, 8, 3.0, seed=125)

    corrector.project(adata_new, batch_key="batch")

    assert "X_test_mnn_corrected" in adata_new.obsm


def test_corrector_project_missing_batch_key_raises() -> None:
    adata = _make_combined_adata(n_batches=2)
    corrector = MNNCorrector(k_mnn=5, verbose=False)
    corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep="X_test",
    )

    adata_new = ad.AnnData(
        X=np.zeros((5, 4)),
        obs=pd.DataFrame(index=[f"new_{i}" for i in range(5)]),
    )
    adata_new.obsm["X_test"] = _make_batch(5, 8, 2.0, seed=124)

    with pytest.raises(KeyError, match="batch_key"):
        corrector.project(adata_new, batch_key="batch")


def test_corrector_correct_requires_same_cells() -> None:
    adata = _make_combined_adata(n_batches=2)
    corrector = MNNCorrector(k_mnn=5, verbose=False)
    corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep="X_test",
    )

    adata_other = _make_combined_adata(n_batches=2, seed=99)
    with pytest.raises(ValueError, match="same cells"):
        corrector.correct(adata_other)


def test_corrector_sparse_pca_fit_and_correct() -> None:
    adata = _make_sparse_combined_adata(n_batches=2)
    corrector = MNNCorrector(k_mnn=5, k_propagate=10, verbose=False)

    corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep=None,
    )
    corrector.correct(adata)

    assert not corrector.store_for_projection
    assert corrector.projection_data_ == {}
    assert "X_pca_mnn_corrected" in adata.obsm
    assert adata.obsm["X_pca_mnn_corrected"].shape[0] == adata.n_obs


def test_corrector_sparse_pca_project_raises() -> None:
    adata = _make_sparse_combined_adata(n_batches=2)
    corrector = MNNCorrector(
        k_mnn=5,
        k_propagate=10,
        store_for_projection=True,
        verbose=False,
    )
    corrector.fit(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep=None,
    )

    _, adata_new = _make_sparse_adata_pair(n_ref=20, n_query=20, d=12, shift=1.0, seed=22)
    adata_new.obs["batch"] = "batch1"
    with pytest.raises(ValueError, match="Projection is unavailable"):
        corrector.project(adata_new, batch_key="batch")


def test_mnn_correct_returns_fitted_corrector() -> None:
    adata_ref, adata_query = _make_adata_pair()

    corrector = mnn_correct_fn(
        adata_ref,
        adata_query,
        use_rep="X_test",
        k_mnn=5,
        k_propagate=10,
        batch_label="query_batch",
        verbose=False,
    )

    assert isinstance(corrector, MNNCorrector)
    assert corrector.is_fitted
    assert "X_test_mnn_corrected" in adata_query.obsm
    assert "query_batch" in corrector.projection_data_


def test_mnn_correct_query_shift_reduced() -> None:
    adata_ref, adata_query = _make_adata_pair(shift=3.0)

    mnn_correct_fn(
        adata_ref,
        adata_query,
        use_rep="X_test",
        k_mnn=5,
        k_propagate=10,
        verbose=False,
    )

    ref_center = adata_ref.obsm["X_test"].mean(axis=0)
    before = adata_query.obsm["X_test"].mean(axis=0)
    after = adata_query.obsm["X_test_mnn_corrected"].mean(axis=0)

    assert np.linalg.norm(after - ref_center) < np.linalg.norm(before - ref_center)


def test_mnn_correct_ref_unchanged() -> None:
    adata_ref, adata_query = _make_adata_pair()
    ref_before = adata_ref.obsm["X_test"].copy()

    mnn_correct_fn(
        adata_ref,
        adata_query,
        use_rep="X_test",
        k_mnn=5,
        k_propagate=10,
        verbose=False,
    )

    np.testing.assert_array_equal(adata_ref.obsm["X_test"], ref_before)


def test_mnn_correct_missing_use_rep_raises() -> None:
    adata_ref, adata_query = _make_adata_pair()
    with pytest.raises(KeyError, match="use_rep"):
        mnn_correct_fn(adata_ref, adata_query, use_rep="X_missing", verbose=False)


def test_mnn_correct_sparse_pca_path() -> None:
    adata_ref, adata_query = _make_sparse_adata_pair()

    corrector = mnn_correct_fn(
        adata_ref,
        adata_query,
        use_rep=None,
        k_mnn=5,
        k_propagate=10,
        verbose=False,
    )

    assert corrector.is_fitted
    assert not corrector.store_for_projection
    assert corrector.projection_data_ == {}
    assert "X_pca_mnn_corrected" in adata_query.obsm


def test_mnn_correct_adata_returns_corrector() -> None:
    adata = _make_combined_adata(n_batches=3)

    result, corrector = mnn_correct_adata(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1", "batch2"],
        use_rep="X_test",
        k_mnn=5,
        k_propagate=10,
        verbose=False,
    )

    assert result is None
    assert isinstance(corrector, MNNCorrector)
    assert corrector.n_corrections_ == 2
    assert "X_test_mnn_corrected" in adata.obsm


def test_mnn_correct_adata_copy() -> None:
    adata = _make_combined_adata(n_batches=2)

    adata_out, corrector = mnn_correct_adata(
        adata,
        batch_key="batch",
        batch_order=["batch0", "batch1"],
        use_rep="X_test",
        k_mnn=5,
        k_propagate=10,
        copy=True,
        verbose=False,
    )

    assert isinstance(corrector, MNNCorrector)
    assert adata_out is not None
    assert adata_out is not adata
    assert "X_test_mnn_corrected" not in adata.obsm
    assert "X_test_mnn_corrected" in adata_out.obsm


def test_mnn_correct_adata_both_order_and_ref_raises() -> None:
    adata = _make_combined_adata(n_batches=2)

    with pytest.raises(ValueError, match="not both"):
        mnn_correct_adata(
            adata,
            batch_key="batch",
            batch_order=["batch0", "batch1"],
            reference="batch0",
            use_rep="X_test",
            verbose=False,
        )


def test_mnn_correct_adata_missing_batch_key_raises() -> None:
    adata = _make_combined_adata(n_batches=2)

    with pytest.raises(KeyError, match="batch_key"):
        mnn_correct_adata(adata, batch_key="missing", use_rep="X_test", verbose=False)


def test_mnn_correct_adata_missing_reference_raises() -> None:
    adata = _make_combined_adata(n_batches=2)

    with pytest.raises(ValueError, match="reference"):
        mnn_correct_adata(
            adata,
            batch_key="batch",
            reference="ghost_batch",
            use_rep="X_test",
            verbose=False,
        )

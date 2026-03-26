"""MNN-based batch correction for AnnData objects."""

from __future__ import annotations

import warnings
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import diags

from .util import wknn

WeightingScheme = Literal["n", "top_n", "jaccard", "jaccard_square", "gaussian", "dist"]


# ──────────────────────────────────────────────────────────────────────────── #
# Internal helper: joint PCA fallback
# ──────────────────────────────────────────────────────────────────────────── #

def _joint_pca(
    X_ref,
    X_query,
    n_components: int = 50,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate raw features from ref and query, run PCA, return both embeddings."""
    import scipy.sparse as sp
    from sklearn.decomposition import PCA

    if verbose:
        print(
            f"[mnn_correct] use_rep=None — running joint PCA "
            f"(n_components={n_components}) on concatenated data..."
        )

    def _dense(X):
        return X.toarray() if sp.issparse(X) else np.asarray(X)

    X_all = np.vstack([_dense(X_ref), _dense(X_query)])
    n_comp = min(n_components, X_all.shape[0] - 1, X_all.shape[1])
    X_pca = PCA(n_components=n_comp).fit_transform(X_all)

    n_ref = X_ref.shape[0]
    return X_pca[:n_ref], X_pca[n_ref:]


# ──────────────────────────────────────────────────────────────────────────── #
# Internal helper: displacement propagation
# ──────────────────────────────────────────────────────────────────────────── #

def _propagate_weighted(
    emb_new: np.ndarray,
    ref_emb: np.ndarray,
    ref_disp: np.ndarray,
    k: int,
    weighting_scheme: WeightingScheme,
    nogpu: bool,
    verbose: bool,
) -> np.ndarray:
    """Propagate *ref_disp* to *emb_new* via a weighted KNN graph.

    Builds a KNN graph from *emb_new* (query) to *ref_emb* (reference),
    row-normalises the weights, and returns the weighted-average displacement
    for every new cell.

    Parameters
    ----------
    emb_new
        Cells to receive propagated displacement, shape ``(n_new, d)``.
    ref_emb
        Reference cells whose displacements are known, shape ``(n_ref, d)``.
    ref_disp
        Known displacements at reference cells, shape ``(n_ref, d)``.
    k, weighting_scheme, nogpu, verbose
        Passed directly to :func:`~mnn_correct.util.wknn.get_wknn`.

    Returns
    -------
    np.ndarray
        Propagated displacement for *emb_new*, shape ``(n_new, d)``.
    """
    k_eff = min(k, ref_emb.shape[0])
    wknn_prop = wknn.get_wknn(
        ref=ref_emb,
        query=emb_new,
        k=k_eff,
        query2ref=True,
        ref2query=False,
        weighting_scheme=weighting_scheme,
        nogpu=nogpu,
        verbose=verbose,
    )
    row_sums = np.array(wknn_prop.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # isolated cells receive zero displacement
    return diags(1.0 / row_sums).dot(wknn_prop).dot(ref_disp)


# ──────────────────────────────────────────────────────────────────────────── #
# MNNCorrector — stateful correction model
# ──────────────────────────────────────────────────────────────────────────── #

class MNNCorrector:
    """Stateful MNN-based batch correction model.

    Estimates per-cell correction displacements from MNN pairs between one or
    more query batches and the reference, and — when
    ``store_for_projection=True`` — can project any new query cells onto a
    previously fitted batch without re-fitting.

    Parameters
    ----------
    k_mnn
        Number of nearest neighbours used when identifying MNN pairs.
    k_propagate
        Number of nearest neighbours used when propagating displacement from
        anchor cells (those with MNNs) to all query cells.
    weighting_scheme
        Edge-weighting scheme for the propagation graph.
    store_for_projection
        If ``True``, :meth:`fit` additionally computes and saves the
        *propagated* displacement for **all** query cells together with their
        raw embeddings, keyed by *batch_label*.  This enables projecting new
        (unseen) cells onto a fitted batch via :meth:`transform` later.
    nogpu
        Force CPU-based neighbour search.
    verbose
        Print progress messages.

    Attributes (available after :meth:`fit`)
    -----------------------------------------
    emb_query_with_mnn_ : np.ndarray, shape (n_anchors, d)
        Latent embeddings of query anchor cells (those with ≥1 MNN) from the
        most recently fitted batch.
    avg_disp_ : np.ndarray, shape (n_anchors, d)
        Per-anchor average MNN displacement from the most recently fitted batch.
    obs_names_with_mnn_ : pd.Index
        Observation names of anchor cells from the most recently fitted batch.
    n_mnn_pairs_ : int
        Total MNN pairs found in the most recent fit.
    n_query_with_mnn_ : int
        Number of unique query anchor cells in the most recent fit.
    projection_data_ : dict
        Per-batch projection store (only populated when
        ``store_for_projection=True``).  Keys are the *batch_label* strings
        passed to :meth:`fit`; values are dicts with four arrays:

        * ``"emb_anchor"``      — embeddings of MNN-anchor cells ``(n_anchors, d)``
        * ``"disp_anchor"``     — raw MNN displacements of anchor cells ``(n_anchors, d)``
        * ``"emb_all"``         — embeddings of *all* query cells ``(n_all, d)``
        * ``"disp_propagated"`` — smoothed displacement for all query cells ``(n_all, d)``

    Examples
    --------
    >>> corrector = MNNCorrector(k_mnn=10, k_propagate=20, store_for_projection=True)
    >>> corrector.fit(emb_ref, emb_query, obs_names_query=adata_q.obs_names,
    ...               batch_label="batch1")
    >>> # Correct training cells (reuses stored propagation — no extra compute):
    >>> emb_corrected = corrector.fit_transform(
    ...     emb_ref, emb_query, batch_label="batch1")
    >>> # Project new unseen cells from the same batch:
    >>> emb_new_corrected = corrector.transform(
    ...     emb_new, batch="batch1", use_propagated=True)
    """

    def __init__(
        self,
        k_mnn: int = 10,
        k_propagate: int = 20,
        weighting_scheme: WeightingScheme = "jaccard_square",
        store_for_projection: bool = False,
        nogpu: bool = False,
        verbose: bool = True,
    ) -> None:
        self.k_mnn = k_mnn
        self.k_propagate = k_propagate
        self.weighting_scheme = weighting_scheme
        self.store_for_projection = store_for_projection
        self.nogpu = nogpu
        self.verbose = verbose

        # ── State from the most recently called fit() ─────────────────────── #
        self.emb_query_with_mnn_: Optional[np.ndarray] = None
        self.avg_disp_: Optional[np.ndarray] = None
        self.obs_names_with_mnn_: Optional[pd.Index] = None
        self.n_mnn_pairs_: int = 0
        self.n_query_with_mnn_: int = 0

        # ── Per-batch projection store ─────────────────────────────────────── #
        # Only populated when store_for_projection=True.
        # projection_data_[batch_label] = {
        #     "emb_anchor"      : (n_anchors, d)  — MNN-anchor cell embeddings
        #     "disp_anchor"     : (n_anchors, d)  — raw per-anchor displacement
        #     "emb_all"         : (n_all,     d)  — all query cell embeddings
        #     "disp_propagated" : (n_all,     d)  — smoothed displacement
        # }
        self.projection_data_: Dict[str, Dict[str, np.ndarray]] = {}

    @property
    def is_fitted(self) -> bool:
        """``True`` after :meth:`fit` has been called at least once."""
        return self.emb_query_with_mnn_ is not None

    def fit(
        self,
        emb_ref: np.ndarray,
        emb_query: np.ndarray,
        obs_names_query: Optional[pd.Index] = None,
        batch_label: str = "default",
    ) -> "MNNCorrector":
        """Estimate per-cell correction displacements from MNN pairs.

        When ``store_for_projection=True`` this also propagates the
        per-anchor displacement to *all* query cells and saves both the raw
        and propagated results in :attr:`projection_data_` under *batch_label*.

        Parameters
        ----------
        emb_ref
            Reference embeddings, shape ``(n_ref, d)``.
        emb_query
            Query embeddings, shape ``(n_query, d)``.
        obs_names_query
            Observation names for query cells.  Defaults to ``RangeIndex``.
        batch_label
            Identifier for this query batch.  Used as the key in
            :attr:`projection_data_` when ``store_for_projection=True``.

        Returns
        -------
        self
        """
        if obs_names_query is None:
            obs_names_query = pd.RangeIndex(emb_query.shape[0])

        if self.verbose:
            print(
                f"[MNNCorrector.fit] Finding MNN pairs (k_mnn={self.k_mnn}) "
                f"between {emb_query.shape[0]:,} query and "
                f"{emb_ref.shape[0]:,} reference cells "
                f"(batch='{batch_label}')..."
            )

        nn_q2r = wknn.build_nn(
            ref=emb_ref, query=emb_query, k=self.k_mnn,
            weight="unweighted", nogpu=self.nogpu, verbose=self.verbose,
        )
        nn_r2q = wknn.build_nn(
            ref=emb_query, query=emb_ref, k=self.k_mnn,
            weight="unweighted", nogpu=self.nogpu, verbose=self.verbose,
        )

        # Mutual: both cells appear in each other's k-NN list
        mnn_matrix = (nn_q2r + nn_r2q.T) == 2  # (n_query, n_ref)
        idx_i, idx_j = mnn_matrix.nonzero()

        if len(idx_i) == 0:
            raise ValueError(
                f"No MNN pairs found with k_mnn={self.k_mnn}. "
                "Try increasing k_mnn or verifying that the two batches share "
                "a common signal in the chosen representation."
            )

        self.n_mnn_pairs_ = len(idx_i)

        # Displacement direction: ref − query  (positive = toward reference)
        displacement_pairs = emb_ref[idx_j] - emb_query[idx_i]
        avg_disp_df = (
            pd.concat(
                [pd.Series(idx_i, name="i"), pd.DataFrame(displacement_pairs)],
                axis=1,
            )
            .groupby("i")
            .mean()
        )
        # groupby sorts by key ⟹ index is sorted unique values from idx_i
        unique_idx_i = avg_disp_df.index.to_numpy()
        avg_disp_df.index = obs_names_query[unique_idx_i]

        self.obs_names_with_mnn_ = avg_disp_df.index
        self.emb_query_with_mnn_ = emb_query[unique_idx_i]
        self.avg_disp_ = avg_disp_df.values
        self.n_query_with_mnn_ = len(unique_idx_i)

        if self.verbose:
            print(
                f"[MNNCorrector.fit] Found {self.n_mnn_pairs_:,} MNN pairs "
                f"covering {self.n_query_with_mnn_:,} query anchor cells."
            )

        # ── Projection store ───────────────────────────────────────────────── #
        if self.store_for_projection:
            if self.verbose:
                print(
                    f"[MNNCorrector.fit] Computing propagated displacement for all "
                    f"{emb_query.shape[0]:,} query cells and storing under "
                    f"batch='{batch_label}'..."
                )
            disp_propagated = _propagate_weighted(
                emb_new=emb_query,
                ref_emb=self.emb_query_with_mnn_,
                ref_disp=self.avg_disp_,
                k=self.k_propagate,
                weighting_scheme=self.weighting_scheme,
                nogpu=self.nogpu,
                verbose=self.verbose,
            )
            self.projection_data_[batch_label] = {
                "emb_anchor":      self.emb_query_with_mnn_.copy(),
                "disp_anchor":     self.avg_disp_.copy(),
                "emb_all":         emb_query.copy(),
                "disp_propagated": disp_propagated,
            }

        return self

    def transform(
        self,
        emb_new: np.ndarray,
        batch: Optional[str] = None,
        use_propagated: bool = False,
    ) -> np.ndarray:
        """Propagate correction displacement to *emb_new* and return corrected embeddings.

        **Default behaviour** (``batch=None``):
            KNN is built from *emb_new* to the MNN-anchor cells of the most
            recently fitted batch; their raw ``avg_disp_`` vectors are
            propagated to every new cell.

        **Batch-specific projection** (``batch=<label>``, requires
        ``store_for_projection=True``):
            KNN is built from *emb_new* to the stored cells of the named batch.
            Two reference sets are supported:

            * ``use_propagated=False`` (default): anchor cells +
              ``disp_anchor`` — tighter reference, only MNN cells.
            * ``use_propagated=True``: all known query cells +
              ``disp_propagated`` — denser reference, smoother result.

        Parameters
        ----------
        emb_new
            Embeddings to correct, shape ``(n_new, d)``.
        batch
            Batch label to look up in :attr:`projection_data_`.  ``None``
            falls back to the default anchor-based propagation.
        use_propagated
            When ``batch`` is given: if ``True`` use the full set of known
            query cells and their propagated displacement as the KNN reference;
            if ``False`` use only MNN-anchor cells and their raw displacement.

        Returns
        -------
        np.ndarray
            Corrected embeddings, shape ``(n_new, d)``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If ``batch`` is given but ``store_for_projection=False``.
        KeyError
            If ``batch`` is not found in :attr:`projection_data_`.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "MNNCorrector must be fitted before calling transform(). "
                "Call fit() or fit_transform() first."
            )

        if batch is not None:
            if not self.store_for_projection:
                raise ValueError(
                    "Batch-specific projection requires store_for_projection=True. "
                    "Re-create the corrector with store_for_projection=True and re-fit."
                )
            if batch not in self.projection_data_:
                raise KeyError(
                    f"No projection data stored for batch '{batch}'. "
                    f"Available batches: {sorted(self.projection_data_.keys())}."
                )
            d = self.projection_data_[batch]
            if use_propagated:
                ref_emb  = d["emb_all"]
                ref_disp = d["disp_propagated"]
                ref_desc = f"all cells of batch='{batch}' (propagated displacement)"
            else:
                ref_emb  = d["emb_anchor"]
                ref_disp = d["disp_anchor"]
                ref_desc = f"anchor cells of batch='{batch}' (raw displacement)"
        else:
            ref_emb  = self.emb_query_with_mnn_
            ref_disp = self.avg_disp_
            ref_desc = "anchor cells of most recently fitted batch"

        if self.verbose:
            print(
                f"[MNNCorrector.transform] Projecting {emb_new.shape[0]:,} new cells "
                f"using {ref_desc} "
                f"(k_propagate={min(self.k_propagate, ref_emb.shape[0])}, "
                f"weighting='{self.weighting_scheme}')..."
            )

        displacement = _propagate_weighted(
            emb_new=emb_new,
            ref_emb=ref_emb,
            ref_disp=ref_disp,
            k=self.k_propagate,
            weighting_scheme=self.weighting_scheme,
            nogpu=self.nogpu,
            verbose=self.verbose,
        )
        return emb_new + displacement

    def fit_transform(
        self,
        emb_ref: np.ndarray,
        emb_query: np.ndarray,
        obs_names_query: Optional[pd.Index] = None,
        batch_label: str = "default",
    ) -> np.ndarray:
        """Fit the model and return corrected query embeddings.

        When ``store_for_projection=True`` this is equivalent to calling
        :meth:`fit` followed by :meth:`transform` with ``batch=batch_label``
        and ``use_propagated=True``, ensuring consistent results.
        """
        self.fit(emb_ref, emb_query, obs_names_query, batch_label)
        if self.store_for_projection:
            return self.transform(emb_query, batch=batch_label, use_propagated=True)
        return self.transform(emb_query)


# ──────────────────────────────────────────────────────────────────────────── #
# mnn_correct — pairwise AnnData correction
# ──────────────────────────────────────────────────────────────────────────── #

def mnn_correct(
    adata_ref: AnnData,
    adata_query: AnnData,
    use_rep: Optional[str] = None,
    n_pca_components: int = 50,
    k_mnn: int = 10,
    k_propagate: int = 20,
    weighting_scheme: WeightingScheme = "jaccard_square",
    store_for_projection: bool = False,
    batch_label: Optional[str] = None,
    key_added: Optional[str] = None,
    nogpu: bool = False,
    verbose: bool = True,
) -> MNNCorrector:
    """Correct batch effects between two AnnData objects using MNN.

    The **reference** batch is never modified.  The corrected query embedding
    is written into ``adata_query.obsm[key_added]`` in-place.

    If ``use_rep`` is ``None`` the function falls back to computing a joint PCA
    on the concatenated raw features (``.X``) of both objects and using that as
    the matched latent representation.

    Parameters
    ----------
    adata_ref
        Reference (anchor) AnnData — cells are not moved.
    adata_query
        Query AnnData — cells will be corrected in-place.
    use_rep
        Key present in both ``adata_ref.obsm`` and ``adata_query.obsm`` for the
        shared latent representation.  Pass ``None`` to trigger joint PCA on
        ``.X`` (requires ``sklearn``).
    n_pca_components
        Number of PCA components when ``use_rep=None``.
    k_mnn
        Number of nearest neighbours for MNN detection.
    k_propagate
        Number of nearest neighbours for displacement propagation.
    weighting_scheme
        Edge-weighting scheme for the propagation graph.
    store_for_projection
        If ``True``, the returned :class:`MNNCorrector` stores the per-anchor
        raw displacement and the propagated displacement for *all* query cells,
        enabling later projection of new (unseen) cells via
        :meth:`~MNNCorrector.transform`.
    batch_label
        Key under which projection data is stored inside the corrector when
        ``store_for_projection=True``.  Defaults to ``"default"``.
    key_added
        Key under which the corrected embedding is stored in
        ``adata_query.obsm``.  Defaults to ``"{use_rep}_mnn_corrected"``
        (or ``"X_pca_mnn_corrected"`` when ``use_rep=None``).
    nogpu
        Force CPU-based neighbour search.
    verbose
        Print progress messages.

    Returns
    -------
    MNNCorrector
        A fitted corrector.  When ``store_for_projection=True`` its
        :meth:`~MNNCorrector.transform` method can project new query cells
        using the saved displacement model.

    Raises
    ------
    KeyError
        If ``use_rep`` is not found in ``adata_ref.obsm`` or ``adata_query.obsm``.
    ValueError
        If no MNN pairs can be identified.

    Examples
    --------
    >>> corrector = mnn_correct(adata_ref, adata_query, use_rep="X_scVI",
    ...                         store_for_projection=True, batch_label="query")
    >>> adata_query.obsm["X_scVI_mnn_corrected"]  # corrected embedding
    >>> # Project new cells from the same batch:
    >>> emb_proj = corrector.transform(emb_new, batch="query", use_propagated=True)
    """
    # ── Resolve embeddings ─────────────────────────────────────────────── #
    if use_rep is None:
        if adata_ref.X is None or adata_query.X is None:
            raise ValueError(
                "use_rep=None requires both adata_ref.X and adata_query.X to be "
                "non-None.  Provide a pre-computed representation via use_rep."
            )
        emb_ref, emb_query = _joint_pca(
            adata_ref.X, adata_query.X,
            n_components=n_pca_components, verbose=verbose,
        )
        _key = key_added or "X_pca_mnn_corrected"
    else:
        if use_rep not in adata_ref.obsm:
            raise KeyError(f"use_rep '{use_rep}' not found in adata_ref.obsm.")
        if use_rep not in adata_query.obsm:
            raise KeyError(f"use_rep '{use_rep}' not found in adata_query.obsm.")
        emb_ref = np.asarray(adata_ref.obsm[use_rep])
        emb_query = np.asarray(adata_query.obsm[use_rep])
        _key = key_added or f"{use_rep}_mnn_corrected"

    if verbose:
        print(
            f"[mnn_correct] {adata_query.n_obs:,} query cells ← "
            f"{adata_ref.n_obs:,} reference cells  →  '{_key}'"
        )

    # ── Fit and apply correction ───────────────────────────────────────── #
    corrector = MNNCorrector(
        k_mnn=k_mnn,
        k_propagate=k_propagate,
        weighting_scheme=weighting_scheme,
        store_for_projection=store_for_projection,
        nogpu=nogpu,
        verbose=verbose,
    )
    _batch_label = batch_label or "default"
    emb_corrected = corrector.fit_transform(
        emb_ref, emb_query,
        obs_names_query=adata_query.obs_names,
        batch_label=_batch_label,
    )
    adata_query.obsm[_key] = emb_corrected

    if verbose:
        print(
            f"[mnn_correct] Done.  Corrected embedding stored in "
            f"adata_query.obsm['{_key}']."
        )

    return corrector


# ──────────────────────────────────────────────────────────────────────────── #
# mnn_correct_adata — multi-batch correction on a single AnnData
# ──────────────────────────────────────────────────────────────────────────── #

def mnn_correct_adata(
    adata: AnnData,
    batch_key: str,
    batch_order: Optional[List[str]] = None,
    reference: Optional[str] = None,
    use_rep: Optional[str] = None,
    n_pca_components: int = 50,
    k_mnn: int = 10,
    k_propagate: int = 20,
    weighting_scheme: WeightingScheme = "jaccard_square",
    store_for_projection: bool = False,
    key_added: Optional[str] = None,
    copy: bool = False,
    nogpu: bool = False,
    verbose: bool = True,
) -> Tuple[Optional[AnnData], List[MNNCorrector]]:
    """Multi-batch MNN correction on a single AnnData.

    Two correction strategies are supported:

    **Sequential** (default, or when *batch_order* is given):
        Batches are aligned one at a time.  The first batch in *batch_order*
        is the initial reference; each subsequent batch is the query.  After
        correction the newly corrected batch is merged into the growing
        reference for the next round.  This propagates corrections across
        all batches in a single pass.

    **Fixed-reference** (when *reference* is given):
        Every non-reference batch is independently corrected against the
        single reference batch.  The reference is never altered.  Suitable
        when one batch serves as a canonical atlas.

    Parameters
    ----------
    adata
        Annotated data matrix containing all batches.
    batch_key
        Column in ``adata.obs`` whose values identify batch membership.
    batch_order
        Ordered list of all batch labels for *sequential* correction.  Ignored
        when ``reference`` is provided.  If ``None`` and ``reference`` is also
        ``None``, batches are sorted alphabetically.
    reference
        Label of the fixed reference batch.  All other batches are
        independently aligned to it.  Mutually exclusive with *batch_order*.
    use_rep
        Key in ``adata.obsm`` for the shared latent representation.  Pass
        ``None`` to run a joint PCA across all cells using ``.X``.
    n_pca_components
        PCA dimensionality when ``use_rep=None``.
    k_mnn
        Number of nearest neighbours for MNN detection.
    k_propagate
        Number of nearest neighbours for displacement propagation.
    weighting_scheme
        Edge-weighting for the propagation graph.
    store_for_projection
        If ``True``, every per-batch :class:`MNNCorrector` stores projection
        data (anchor displacements + propagated displacements) keyed by the
        query batch label, enabling later projection of new cells via
        :meth:`~MNNCorrector.transform`.
    key_added
        Key under which corrected embeddings are stored in ``adata.obsm``.
        Defaults to ``"{use_rep}_mnn_corrected"``
        (or ``"X_pca_mnn_corrected"`` when ``use_rep=None``).
        Reference cells are initialised to their original representation and
        are unchanged.
    copy
        Return a corrected copy instead of modifying in-place.
    nogpu
        Force CPU-based neighbour search.
    verbose
        Print progress messages.

    Returns
    -------
    adata : AnnData or None
        The corrected AnnData if ``copy=True``, else ``None``.
    correctors : list[MNNCorrector]
        One fitted :class:`MNNCorrector` per correction round, in order.

    Raises
    ------
    KeyError
        If ``batch_key`` or ``use_rep`` are not found in ``adata``.
    ValueError
        If ``reference`` and ``batch_order`` are both provided, or if a
        batch label cannot be resolved.

    Examples
    --------
    >>> # Sequential (3 batches)
    >>> _, correctors = mnn_correct_adata(
    ...     adata, batch_key="batch", use_rep="X_scVI",
    ...     batch_order=["batch1", "batch2", "batch3"],
    ... )

    >>> # Fixed reference
    >>> _, correctors = mnn_correct_adata(
    ...     adata, batch_key="batch", use_rep="X_scVI", reference="atlas",
    ... )
    """
    # ── Validation ─────────────────────────────────────────────────────── #
    if batch_key not in adata.obs.columns:
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs.")
    if reference is not None and batch_order is not None:
        raise ValueError(
            "Provide either 'reference' (fixed-reference mode) or 'batch_order' "
            "(sequential mode), not both."
        )

    observed: List[str] = sorted(adata.obs[batch_key].unique().tolist())

    if reference is not None and reference not in observed:
        raise ValueError(
            f"reference '{reference}' not found in adata.obs['{batch_key}']. "
            f"Available: {observed}."
        )
    if batch_order is not None:
        missing = set(observed) - set(batch_order)
        extra = set(batch_order) - set(observed)
        if missing:
            raise ValueError(
                f"batch_order is missing labels present in the data: {sorted(missing)}."
            )
        if extra:
            raise ValueError(
                f"batch_order contains labels not found in the data: {sorted(extra)}."
            )

    if copy:
        adata = adata.copy()

    # ── Resolve source representation ──────────────────────────────────── #
    if use_rep is None:
        if adata.X is None:
            raise ValueError("use_rep=None requires adata.X to be non-None.")
        import scipy.sparse as sp
        from sklearn.decomposition import PCA

        if verbose:
            print(
                f"[mnn_correct_adata] use_rep=None — running joint PCA "
                f"(n_components={n_pca_components}) across all {adata.n_obs:,} cells..."
            )
        X_all = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        n_comp = min(n_pca_components, adata.n_obs - 1, adata.n_vars)
        _pca_key = "_X_pca_mnn_joint"
        adata.obsm[_pca_key] = PCA(n_components=n_comp).fit_transform(X_all)
        src_rep = _pca_key
        _key_added = key_added or "X_pca_mnn_corrected"
    else:
        if use_rep not in adata.obsm:
            raise KeyError(f"use_rep '{use_rep}' not found in adata.obsm.")
        src_rep = use_rep
        _key_added = key_added or f"{use_rep}_mnn_corrected"

    # Initialise key_added for all cells as a copy of the source representation.
    # Reference cells will keep these values unchanged.
    adata.obsm[_key_added] = np.asarray(adata.obsm[src_rep]).copy()

    correctors: List[MNNCorrector] = []

    # ── Sequential mode ────────────────────────────────────────────────── #
    if reference is None:
        order: List[str] = batch_order if batch_order is not None else observed
        if verbose:
            print(
                f"[mnn_correct_adata] Sequential mode — "
                f"batch order: {order}"
            )

        for i in range(1, len(order)):
            query_label = order[i]
            ref_labels = order[:i]
            ref_mask = adata.obs[batch_key].isin(ref_labels).values
            query_mask = (adata.obs[batch_key] == query_label).values

            if verbose:
                print(
                    f"\n[mnn_correct_adata] Round {i}/{len(order) - 1}: "
                    f"ref={ref_labels} ({ref_mask.sum():,} cells)  →  "
                    f"query='{query_label}' ({query_mask.sum():,} cells)"
                )

            # Use copies so that obsm assignments inside mnn_correct are safe
            adata_ref_i = adata[ref_mask].copy()
            adata_query_i = adata[query_mask].copy()

            corrector_i = mnn_correct(
                adata_ref_i, adata_query_i,
                use_rep=_key_added,
                k_mnn=k_mnn, k_propagate=k_propagate,
                weighting_scheme=weighting_scheme,
                store_for_projection=store_for_projection,
                batch_label=query_label,
                key_added=_key_added,
                nogpu=nogpu, verbose=verbose,
            )
            correctors.append(corrector_i)

            # Write corrected embeddings back into the main adata
            adata.obsm[_key_added][query_mask] = adata_query_i.obsm[_key_added]

    # ── Fixed-reference mode ───────────────────────────────────────────── #
    else:
        query_labels = [b for b in observed if b != reference]
        if verbose:
            print(
                f"[mnn_correct_adata] Fixed-reference mode — "
                f"reference='{reference}', query batches: {query_labels}"
            )

        ref_mask = (adata.obs[batch_key] == reference).values
        adata_ref_base = adata[ref_mask].copy()

        for qi, query_label in enumerate(query_labels):
            query_mask = (adata.obs[batch_key] == query_label).values

            if verbose:
                print(
                    f"\n[mnn_correct_adata] Round {qi + 1}/{len(query_labels)}: "
                    f"ref='{reference}' ({ref_mask.sum():,} cells)  →  "
                    f"query='{query_label}' ({query_mask.sum():,} cells)"
                )

            adata_query_i = adata[query_mask].copy()

            corrector_i = mnn_correct(
                adata_ref_base, adata_query_i,
                use_rep=_key_added,
                k_mnn=k_mnn, k_propagate=k_propagate,
                weighting_scheme=weighting_scheme,
                store_for_projection=store_for_projection,
                batch_label=query_label,
                key_added=_key_added,
                nogpu=nogpu, verbose=verbose,
            )
            correctors.append(corrector_i)

            adata.obsm[_key_added][query_mask] = adata_query_i.obsm[_key_added]

    if verbose:
        print(
            f"\n[mnn_correct_adata] Complete.  Corrected embedding stored in "
            f"adata.obsm['{_key_added}']."
        )

    return (adata if copy else None), correctors

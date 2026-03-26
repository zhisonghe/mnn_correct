"""MNN-based batch correction for AnnData objects."""

from __future__ import annotations

import warnings
from typing import List, Literal, Optional, Tuple

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
# MNNCorrector — stateful correction model
# ──────────────────────────────────────────────────────────────────────────── #

class MNNCorrector:
    """Stateful MNN-based batch correction model.

    Estimates per-cell correction displacements from MNN pairs between a
    reference and a query batch and stores the anchor information needed to
    project new (unseen) query cells without re-fitting.

    Parameters
    ----------
    k_mnn
        Number of nearest neighbours used when identifying MNN pairs.
    k_propagate
        Number of nearest neighbours used when propagating displacement from
        anchor cells (those with MNNs) to all query cells.
    weighting_scheme
        Edge-weighting scheme for the propagation graph.
    nogpu
        Force CPU-based neighbour search.
    verbose
        Print progress messages.

    Attributes (available after :meth:`fit`)
    -----------------------------------------
    emb_query_with_mnn_ : np.ndarray, shape (n_anchors, d)
        Latent embeddings of query anchor cells (those with ≥1 MNN).
    avg_disp_ : np.ndarray, shape (n_anchors, d)
        Per-anchor-cell average correction displacement.
    obs_names_with_mnn_ : pd.Index
        Original observation names of anchor cells.
    n_mnn_pairs_ : int
        Total number of MNN pairs found.
    n_query_with_mnn_ : int
        Number of unique query cells that have ≥1 MNN partner.

    Examples
    --------
    >>> corrector = MNNCorrector(k_mnn=10, k_propagate=20)
    >>> corrector.fit(emb_ref, emb_query, obs_names_query=adata_query.obs_names)
    >>> emb_corrected = corrector.transform(emb_new)  # project unseen cells
    """

    def __init__(
        self,
        k_mnn: int = 10,
        k_propagate: int = 20,
        weighting_scheme: WeightingScheme = "jaccard_square",
        nogpu: bool = False,
        verbose: bool = True,
    ) -> None:
        self.k_mnn = k_mnn
        self.k_propagate = k_propagate
        self.weighting_scheme = weighting_scheme
        self.nogpu = nogpu
        self.verbose = verbose

        # Fitted state (populated by fit())
        self.emb_query_with_mnn_: Optional[np.ndarray] = None
        self.avg_disp_: Optional[np.ndarray] = None
        self.obs_names_with_mnn_: Optional[pd.Index] = None
        self.n_mnn_pairs_: int = 0
        self.n_query_with_mnn_: int = 0

    @property
    def is_fitted(self) -> bool:
        """``True`` after :meth:`fit` has been called successfully."""
        return self.emb_query_with_mnn_ is not None

    def fit(
        self,
        emb_ref: np.ndarray,
        emb_query: np.ndarray,
        obs_names_query: Optional[pd.Index] = None,
    ) -> "MNNCorrector":
        """Estimate per-cell correction displacements from MNN pairs.

        Parameters
        ----------
        emb_ref
            Reference embeddings, shape ``(n_ref, d)``.
        emb_query
            Query embeddings, shape ``(n_query, d)``.
        obs_names_query
            Observation names for query cells (indexes :attr:`avg_disp_` and
            :attr:`obs_names_with_mnn_`).  Defaults to ``RangeIndex``.

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
                f"{emb_ref.shape[0]:,} reference cells..."
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

        return self

    def transform(
        self,
        emb_query: np.ndarray,
    ) -> np.ndarray:
        """Propagate saved displacement to query embeddings.

        Builds a weighted KNN graph from every query cell to the stored anchor
        cells (:attr:`emb_query_with_mnn_`) and computes a weighted-average
        displacement for each cell.  Can be called on new (unseen) cells as
        long as their embeddings live in the same latent space.

        Parameters
        ----------
        emb_query
            Embeddings to correct, shape ``(n_query, d)``.

        Returns
        -------
        np.ndarray
            Corrected embeddings, shape ``(n_query, d)``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "MNNCorrector must be fitted before calling transform(). "
                "Call fit() or fit_transform() first."
            )

        k = min(self.k_propagate, self.n_query_with_mnn_)

        if self.verbose:
            print(
                f"[MNNCorrector.transform] Propagating displacement to "
                f"{emb_query.shape[0]:,} cells "
                f"(k_propagate={k}, weighting='{self.weighting_scheme}')..."
            )

        wknn_prop = wknn.get_wknn(
            ref=self.emb_query_with_mnn_,
            query=emb_query,
            k=k,
            query2ref=True,
            ref2query=False,
            weighting_scheme=self.weighting_scheme,
            nogpu=self.nogpu,
            verbose=self.verbose,
        )

        row_sums = np.array(wknn_prop.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0  # isolated cells receive zero displacement
        mat_weights = diags(1.0 / row_sums).dot(wknn_prop)

        displacement = mat_weights.dot(self.avg_disp_)
        return emb_query + displacement

    def fit_transform(
        self,
        emb_ref: np.ndarray,
        emb_query: np.ndarray,
        obs_names_query: Optional[pd.Index] = None,
    ) -> np.ndarray:
        """Fit the model and return corrected query embeddings.

        Equivalent to ``fit(emb_ref, emb_query, obs_names_query).transform(emb_query)``.
        """
        self.fit(emb_ref, emb_query, obs_names_query)
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
        A fitted corrector whose :meth:`~MNNCorrector.transform` method can
        project new (unseen) query cells using the saved displacement model.

    Raises
    ------
    KeyError
        If ``use_rep`` is not found in ``adata_ref.obsm`` or ``adata_query.obsm``.
    ValueError
        If no MNN pairs can be identified.

    Examples
    --------
    >>> corrector = mnn_correct(adata_ref, adata_query, use_rep="X_scVI")
    >>> adata_query.obsm["X_scVI_mnn_corrected"]  # corrected embedding
    >>> emb_new_corrected = corrector.transform(emb_new)  # project new cells
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
        nogpu=nogpu,
        verbose=verbose,
    )
    emb_corrected = corrector.fit_transform(
        emb_ref, emb_query, obs_names_query=adata_query.obs_names,
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

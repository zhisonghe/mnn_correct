"""MNN-based batch correction for AnnData objects."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData, concat

from .util import NeighborFlavor, WeightingScheme, propagate_weighted, wknn


class MNNCorrector:
    """Stateful MNN batch-correction model for AnnData workflows.

    The main workflow is:

    1. :meth:`fit` estimates per-batch displacement models from a training
       :class:`~anndata.AnnData` object.
    2. :meth:`correct` writes the fitted correction back into an AnnData object
       containing the same cells used during fitting.
    3. :meth:`project` propagates a previously estimated batch displacement to
       new cells that belong to an already fitted batch label.

    Parameters
    ----------
    k_mnn
        Number of nearest neighbours used when detecting mutual nearest
        neighbour pairs.
    k_propagate
        Number of neighbours used when propagating anchor-cell displacements
        to the full query batch.
    weighting_scheme
        Weighting scheme passed to the weighted KNN propagation graph.
    store_for_projection
        Whether to retain per-batch projection state after fitting. When
        ``use_rep=None``, this is forced to ``False`` because the PCA fallback
        only supports the current correction run.
    flavor
        Neighbor-search backend. Use ``"auto"`` to select scikit-learn for
        small inputs, cuML for large GPU-enabled inputs, and pynndescent
        otherwise.
    verbose
        If ``True``, print progress messages during fitting and projection.

    Attributes
    ----------
    projection_data_
        Mapping from fitted batch label to stored anchor embeddings,
        propagated displacements, and corrected embeddings used for
        projection.
    key_added_
        Output key used by :meth:`correct` and :meth:`project` when writing
        corrected embeddings into ``adata.obsm``.
    n_corrections_
        Number of correction rounds estimated during the most recent fit.
    """

    def __init__(
        self,
        k_mnn: int = 10,
        k_propagate: int = 20,
        weighting_scheme: WeightingScheme = "jaccard_square",
        store_for_projection: bool = True,
        flavor: NeighborFlavor = "auto",
        verbose: bool = True,
    ) -> None:
        """Initialize a stateful MNN corrector."""
        self.k_mnn = k_mnn
        self.k_propagate = k_propagate
        self.weighting_scheme = weighting_scheme
        self._store_for_projection_requested = store_for_projection
        self.store_for_projection = store_for_projection
        self.flavor = flavor
        self.verbose = verbose

        self.emb_query_with_mnn_: Optional[np.ndarray] = None
        self.avg_disp_: Optional[np.ndarray] = None
        self.obs_names_with_mnn_: Optional[pd.Index] = None
        self.n_mnn_pairs_: int = 0
        self.n_query_with_mnn_: int = 0

        self.projection_data_: Dict[str, Dict[str, Any]] = {}
        self.n_corrections_: int = 0

        self.batch_key_: Optional[str] = None
        self.batch_order_: Optional[List[str]] = None
        self.reference_: Optional[str] = None
        self.use_rep_: Optional[str] = None
        self.key_added_: Optional[str] = None
        self.fitted_obs_names_: Optional[pd.Index] = None
        self.fitted_batches_: List[str] = []

        self._corrected_embeddings_: Optional[np.ndarray] = None
        self._source_embeddings_: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has estimated correction state.

        Returns
        -------
        bool
            ``True`` once correction and projection state are available.
        """
        return self._corrected_embeddings_ is not None

    def _reset_fit_state(self) -> None:
        """Clear all state associated with a previous fit."""
        self.emb_query_with_mnn_ = None
        self.avg_disp_ = None
        self.obs_names_with_mnn_ = None
        self.n_mnn_pairs_ = 0
        self.n_query_with_mnn_ = 0
        self.projection_data_ = {}
        self.n_corrections_ = 0
        self.batch_key_ = None
        self.batch_order_ = None
        self.reference_ = None
        self.use_rep_ = None
        self.key_added_ = None
        self.fitted_obs_names_ = None
        self.fitted_batches_ = []
        self._corrected_embeddings_ = None
        self._source_embeddings_ = None

    def _set_last_batch_state(self, batch_model: Dict[str, Any]) -> None:
        """Expose the most recently fitted batch model through legacy attributes.

        Parameters
        ----------
        batch_model
            Internal batch model dictionary produced by
            :meth:`_estimate_batch_model`.
        """
        self.emb_query_with_mnn_ = np.asarray(batch_model["emb_anchor"])
        self.avg_disp_ = np.asarray(batch_model["disp_anchor"])
        self.obs_names_with_mnn_ = pd.Index(batch_model["obs_names_anchor"])
        self.n_mnn_pairs_ = int(batch_model["n_mnn_pairs"])
        self.n_query_with_mnn_ = int(batch_model["n_query_with_mnn"])

    def _estimate_batch_model(
        self,
        emb_ref: np.ndarray,
        emb_query: np.ndarray,
        obs_names_query: pd.Index,
        batch_label: str,
    ) -> Dict[str, Any]:
        """Estimate correction state for a single query batch against a reference.

        Parameters
        ----------
        emb_ref
            Corrected embedding for the current reference cells.
        emb_query
            Embedding for the current query batch.
        obs_names_query
            Observation names aligned to ``emb_query``.
        batch_label
            Label used to store this fitted batch model.

        Returns
        -------
        dict
            Stored batch model containing anchor embeddings, displacement
            vectors, propagated correction for all query cells, and summary
            counts.
        """
        if self.verbose:
            print(
                f"[MNNCorrector.fit] Finding MNN pairs (k_mnn={self.k_mnn}) "
                f"between {emb_query.shape[0]:,} query and "
                f"{emb_ref.shape[0]:,} reference cells "
                f"(batch='{batch_label}')..."
            )

        mnn_matrix = wknn.build_mutual_nn(
            dat1=emb_query,
            dat2=emb_ref,
            k1=self.k_mnn,
            k2=self.k_mnn,
            flavor=self.flavor,
            verbose=self.verbose,
        )
        idx_i, idx_j = mnn_matrix.nonzero()
        if len(idx_i) == 0:
            raise ValueError(
                f"No MNN pairs found with k_mnn={self.k_mnn} for batch '{batch_label}'. "
                "Try increasing k_mnn or verifying that the batches share a "
                "common signal in the chosen representation."
            )

        displacement_pairs = emb_ref[idx_j] - emb_query[idx_i]
        avg_disp_df = (
            pd.concat(
                [pd.Series(idx_i, name="i"), pd.DataFrame(displacement_pairs)],
                axis=1,
            )
            .groupby("i")
            .mean()
        )
        unique_idx_i = avg_disp_df.index.to_numpy()
        avg_disp_df.index = obs_names_query[unique_idx_i]

        emb_anchor = emb_query[unique_idx_i]
        disp_anchor = avg_disp_df.to_numpy()
        disp_propagated = np.asarray(
            propagate_weighted(
                emb_new=emb_query,
                ref_emb=emb_anchor,
                ref_disp=disp_anchor,
                k=self.k_propagate,
                weighting_scheme=self.weighting_scheme,
                flavor=self.flavor,
                verbose=self.verbose,
            )
        )

        if self.verbose:
            print(
                f"[MNNCorrector.fit] Found {len(idx_i):,} MNN pairs covering "
                f"{len(unique_idx_i):,} query anchor cells for batch '{batch_label}'."
            )

        return {
            "batch_label": batch_label,
            "obs_names_all": pd.Index(obs_names_query),
            "obs_names_anchor": avg_disp_df.index.copy(),
            "emb_anchor": emb_anchor.copy(),
            "disp_anchor": disp_anchor.copy(),
            "emb_all": emb_query.copy(),
            "disp_propagated": disp_propagated,
            "corrected_all": emb_query + disp_propagated,
            "n_mnn_pairs": len(idx_i),
            "n_query_with_mnn": len(unique_idx_i),
        }

    def _resolve_fit_representation(
        self,
        adata: AnnData,
        use_rep: Optional[str],
        n_pca_components: int,
    ) -> Tuple[np.ndarray, str]:
        """Resolve the representation used for fitting.

        Parameters
        ----------
        adata
            Training AnnData object.
        use_rep
            Key in ``adata.obsm`` to use as the source embedding, or ``None``
            to fit a PCA model on ``adata.X`` with :func:`scanpy.pp.pca` for
            the current correction run only.
        n_pca_components
            Number of PCA components when ``use_rep=None``.

        Returns
        -------
        tuple[numpy.ndarray, str]
            Dense embedding matrix and the default output key for corrected
            embeddings.
        """
        if use_rep is None:
            if adata.X is None:
                raise ValueError("use_rep=None requires adata.X to be non-None.")

            if self.verbose:
                print(
                    f"[MNNCorrector.fit] use_rep=None -- running scanpy.pp.pca "
                    f"(n_components={n_pca_components}) across all {adata.n_obs:,} cells..."
                )

            n_comp = min(n_pca_components, min(adata.n_obs, adata.n_vars) - 1)
            if n_comp < 1:
                raise ValueError("PCA requires at least two cells and one feature.")

            pca_adata = AnnData(X=adata.X.copy())
            sc.pp.pca(pca_adata, n_comps=n_comp, dtype="float64")
            embeddings = np.asarray(pca_adata.obsm["X_pca"])
            return embeddings, "X_pca_mnn_corrected"

        if use_rep not in adata.obsm:
            raise KeyError(f"use_rep '{use_rep}' not found in adata.obsm.")

        return np.asarray(adata.obsm[use_rep]), f"{use_rep}_mnn_corrected"

    def _resolve_project_representation(
        self,
        adata: AnnData,
        use_rep: Optional[str],
    ) -> np.ndarray:
        """Resolve the source representation used for projection or validation.

        Parameters
        ----------
        adata
            AnnData object whose cells should be projected or validated.
        use_rep
            Optional representation override. This must match the fitted
            representation when the corrector was not trained from ``adata.X``.

        Returns
        -------
        numpy.ndarray
            Dense embedding matrix used for projection.
        """
        if self.use_rep_ is None:
            raise ValueError(
                "Projection is unavailable when fit() was run with use_rep=None. "
                "Provide a reusable representation in adata.obsm to enable project()."
            )

        rep_key = use_rep or self.use_rep_
        if rep_key != self.use_rep_:
            raise ValueError(
                f"project() expected use_rep='{self.use_rep_}', got '{rep_key}'."
            )
        if rep_key not in adata.obsm:
            raise KeyError(f"use_rep '{rep_key}' not found in adata.obsm.")
        return np.asarray(adata.obsm[rep_key])

    def _project_embeddings(
        self,
        emb_new: np.ndarray,
        batch_label: str,
        use_propagated: bool,
    ) -> np.ndarray:
        """Project stored displacement vectors onto a raw embedding matrix.

        Parameters
        ----------
        emb_new
            Embeddings to correct.
        batch_label
            Previously fitted batch label whose projection model should be used.
        use_propagated
            If ``True``, use propagated displacement from all fitted cells in
            the batch. If ``False``, use only anchor-cell displacement.

        Returns
        -------
        numpy.ndarray
            Corrected embedding matrix.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "MNNCorrector must be fitted before calling project(). Call fit() first."
            )
        if not self.store_for_projection:
            raise ValueError(
                "Projection is unavailable for this fitted corrector. Fit with a reusable "
                "representation in adata.obsm and store_for_projection=True."
            )
        if batch_label not in self.projection_data_:
            raise KeyError(
                f"No projection data stored for batch '{batch_label}'. "
                f"Available batches: {sorted(self.projection_data_.keys())}."
            )

        batch_model = self.projection_data_[batch_label]
        if use_propagated:
            ref_emb = np.asarray(batch_model["emb_all"])
            ref_disp = np.asarray(batch_model["disp_propagated"])
            ref_desc = f"all fitted cells from batch='{batch_label}'"
        else:
            ref_emb = np.asarray(batch_model["emb_anchor"])
            ref_disp = np.asarray(batch_model["disp_anchor"])
            ref_desc = f"anchor cells from batch='{batch_label}'"

        if self.verbose:
            print(
                f"[MNNCorrector.project] Projecting {emb_new.shape[0]:,} cells using {ref_desc} "
                f"(k_propagate={min(self.k_propagate, ref_emb.shape[0])}, "
                f"weighting='{self.weighting_scheme}')..."
            )

        displacement = propagate_weighted(
            emb_new=emb_new,
            ref_emb=ref_emb,
            ref_disp=ref_disp,
            k=self.k_propagate,
            weighting_scheme=self.weighting_scheme,
            flavor=self.flavor,
            verbose=self.verbose,
        )
        return emb_new + np.asarray(displacement)

    def fit(
        self,
        adata: AnnData,
        batch_key: str,
        batch_order: Optional[List[str]] = None,
        reference: Optional[str] = None,
        use_rep: Optional[str] = None,
        n_pca_components: int = 50,
        key_added: Optional[str] = None,
        return_corrector: bool = False,
    ) -> Optional["MNNCorrector"]:
        """Estimate and store per-batch displacement models from an AnnData object.

        Parameters
        ----------
        adata
            AnnData object containing all batches involved in model fitting.
        batch_key
            Column in ``adata.obs`` identifying batch membership.
        batch_order
            Optional sequential correction order. The first batch is treated as
            the initial reference and each subsequent batch is corrected in
            turn against the growing reference.
        reference
            Optional fixed reference batch. When provided, every other batch is
            corrected independently against this batch.
        use_rep
            Key in ``adata.obsm`` containing the latent representation to
            correct. If ``None``, PCA is fit on ``adata.X`` for the current
            correction only and future projection is disabled.
        n_pca_components
            Number of PCA components when ``use_rep=None``.
        key_added
            Optional output key for corrected embeddings written by
            :meth:`correct` and :meth:`project`.
        return_corrector
            If ``True``, return the fitted corrector instance after updating
            its internal state. By default, :meth:`fit` only updates the
            current object and returns ``None``.

        Returns
        -------
        MNNCorrector or None
            The fitted corrector instance when ``return_corrector=True``;
            otherwise ``None``.

        Raises
        ------
        KeyError
            If ``batch_key`` or ``use_rep`` cannot be resolved.
        ValueError
            If both ``batch_order`` and ``reference`` are provided or if the
            requested batch labels are inconsistent with the input data.
        """
        if batch_key not in adata.obs.columns:
            raise KeyError(f"batch_key '{batch_key}' not found in adata.obs.")
        if reference is not None and batch_order is not None:
            raise ValueError(
                "Provide either 'reference' (fixed-reference mode) or 'batch_order' "
                "(sequential mode), not both."
            )

        observed = sorted(adata.obs[batch_key].unique().tolist())
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

        self._reset_fit_state()
        source_embeddings, default_key_added = self._resolve_fit_representation(
            adata, use_rep, n_pca_components
        )
        self.store_for_projection = self._store_for_projection_requested
        if use_rep is None:
            self.store_for_projection = False
            if self.verbose and self._store_for_projection_requested:
                print(
                    "[MNNCorrector.fit] use_rep=None disables future projection; "
                    "store_for_projection has been forced to False for this fit."
                )
        corrected_embeddings = source_embeddings.copy()
        batch_values = adata.obs[batch_key]

        self.batch_key_ = batch_key
        self.reference_ = reference
        self.use_rep_ = use_rep
        self.key_added_ = key_added or default_key_added
        self.fitted_obs_names_ = adata.obs_names.copy()

        if reference is None:
            order = batch_order if batch_order is not None else observed
            self.batch_order_ = list(order)
            if self.verbose:
                print(f"[MNNCorrector.fit] Sequential mode -- batch order: {order}")

            for index in range(1, len(order)):
                query_label = order[index]
                ref_labels = order[:index]
                ref_mask = batch_values.isin(ref_labels).to_numpy()
                query_mask = (batch_values == query_label).to_numpy()

                if self.verbose:
                    print(
                        f"[MNNCorrector.fit] Round {index}/{len(order) - 1}: "
                        f"ref={ref_labels} ({int(ref_mask.sum()):,} cells) -> "
                        f"query='{query_label}' ({int(query_mask.sum()):,} cells)"
                    )

                batch_model = self._estimate_batch_model(
                    emb_ref=corrected_embeddings[ref_mask],
                    emb_query=corrected_embeddings[query_mask],
                    obs_names_query=adata.obs_names[query_mask],
                    batch_label=query_label,
                )
                corrected_embeddings[query_mask] = np.asarray(batch_model["corrected_all"])
                if self.store_for_projection:
                    self.projection_data_[query_label] = batch_model
                self.fitted_batches_.append(query_label)
                self._set_last_batch_state(batch_model)
        else:
            ref_mask = (batch_values == reference).to_numpy()
            query_labels = [label for label in observed if label != reference]
            if self.verbose:
                print(
                    f"[MNNCorrector.fit] Fixed-reference mode -- reference='{reference}', "
                    f"query batches: {query_labels}"
                )

            for index, query_label in enumerate(query_labels, start=1):
                query_mask = (batch_values == query_label).to_numpy()

                if self.verbose:
                    print(
                        f"[MNNCorrector.fit] Round {index}/{len(query_labels)}: "
                        f"ref='{reference}' ({int(ref_mask.sum()):,} cells) -> "
                        f"query='{query_label}' ({int(query_mask.sum()):,} cells)"
                    )

                batch_model = self._estimate_batch_model(
                    emb_ref=corrected_embeddings[ref_mask],
                    emb_query=corrected_embeddings[query_mask],
                    obs_names_query=adata.obs_names[query_mask],
                    batch_label=query_label,
                )
                corrected_embeddings[query_mask] = np.asarray(batch_model["corrected_all"])
                if self.store_for_projection:
                    self.projection_data_[query_label] = batch_model
                self.fitted_batches_.append(query_label)
                self._set_last_batch_state(batch_model)

        self._corrected_embeddings_ = corrected_embeddings
        self._source_embeddings_ = source_embeddings.copy()
        self.n_corrections_ = len(self.fitted_batches_)

        if self.verbose:
            print(
                f"[MNNCorrector.fit] Stored displacement models for "
                f"{self.n_corrections_:,} batch correction round(s)."
            )

        return self if return_corrector else None

    def correct(
        self,
        adata: AnnData,
        key_added: Optional[str] = None,
        copy: bool = False,
    ) -> Optional[AnnData]:
        """Apply the fitted correction to the AnnData used during :meth:`fit`.

        Parameters
        ----------
        adata
            The same AnnData object, or a reordered view containing the same
            cells and source representation, that was used during fitting.
        key_added
            Optional override for the output key in ``adata.obsm``.
        copy
            If ``True``, return a corrected copy instead of modifying ``adata``
            in place.

        Returns
        -------
        AnnData or None
            Corrected copy when ``copy=True``; otherwise ``None``.

        Raises
        ------
        RuntimeError
            If the corrector has not been fitted.
        ValueError
            If ``adata`` does not match the cells and representation used
            during fitting.
        """
        if (
            not self.is_fitted
            or self.fitted_obs_names_ is None
            or self._corrected_embeddings_ is None
            or self._source_embeddings_ is None
        ):
            raise RuntimeError(
                "MNNCorrector must be fitted before calling correct(). Call fit() first."
            )

        indexer = self.fitted_obs_names_.get_indexer(adata.obs_names)
        if len(adata.obs_names) != len(self.fitted_obs_names_) or np.any(indexer < 0):
            raise ValueError(
                "correct() expects the same cells used during fit(). Use project() for new cells."
            )

        if self.use_rep_ is not None:
            current_source = self._resolve_project_representation(adata, self.use_rep_)
            if not np.allclose(current_source, self._source_embeddings_[indexer]):
                raise ValueError(
                    "correct() expects the same cells and source representation used during fit(). "
                    "Use project() for new cells."
                )

        target = adata.copy() if copy else adata
        target_key = key_added or self.key_added_
        if target_key is None:
            raise RuntimeError("No output key is available. Re-fit the corrector first.")

        target.obsm[target_key] = self._corrected_embeddings_[indexer].copy()

        if self.verbose:
            print(
                f"[MNNCorrector.correct] Corrected embedding stored in adata.obsm['{target_key}']."
            )

        return target if copy else None

    def project(
        self,
        adata: AnnData,
        batch_label: str,
        use_rep: Optional[str] = None,
        key_added: Optional[str] = None,
        use_propagated: bool = True,
        copy: bool = False,
    ) -> Optional[AnnData]:
        """Project a fitted batch displacement model onto new cells.

        Parameters
        ----------
        adata
            AnnData containing new cells from a previously fitted batch.
        batch_label
            Batch label whose fitted displacement model should be reused.
        use_rep
            Optional representation key. This must match the representation
            used during fitting.
        key_added
            Optional override for the output key in ``adata.obsm``.
        use_propagated
            If ``True``, project from all fitted cells in the batch using their
            propagated displacements. If ``False``, use only MNN anchor cells.
        copy
            If ``True``, return a corrected copy instead of modifying ``adata``
            in place.

        Returns
        -------
        AnnData or None
            Corrected copy when ``copy=True``; otherwise ``None``.
        """
        emb_new = self._resolve_project_representation(adata, use_rep)
        corrected = self._project_embeddings(emb_new, batch_label, use_propagated)

        target = adata.copy() if copy else adata
        target_key = key_added or self.key_added_
        if target_key is None:
            raise RuntimeError("No output key is available. Re-fit the corrector first.")

        target.obsm[target_key] = corrected
        return target if copy else None

    def transform(
        self,
        emb_new: np.ndarray,
        batch: str,
        use_propagated: bool = True,
    ) -> np.ndarray:
        """Project raw embeddings using a fitted batch model.

        This low-level helper is retained for backwards compatibility. New
        AnnData-based workflows should prefer :meth:`project`.

        Parameters
        ----------
        emb_new
            Embeddings to correct.
        batch
            Fitted batch label whose displacement model should be reused.
        use_propagated
            Whether to project from all fitted cells or anchor cells only.

        Returns
        -------
        numpy.ndarray
            Corrected embedding matrix.
        """
        return self._project_embeddings(emb_new, batch, use_propagated)


def mnn_correct(
    adata_ref: AnnData,
    adata_query: AnnData,
    use_rep: Optional[str] = None,
    n_pca_components: int = 50,
    k_mnn: int = 10,
    k_propagate: int = 20,
    weighting_scheme: WeightingScheme = "jaccard_square",
    store_for_projection: bool = True,
    batch_label: Optional[str] = None,
    key_added: Optional[str] = None,
    flavor: NeighborFlavor = "auto",
    verbose: bool = True,
) -> MNNCorrector:
    """Correct batch effects between two AnnData objects using MNN.

    Parameters
    ----------
    adata_ref
        Reference AnnData. Its embeddings are not modified.
    adata_query
        Query AnnData. The corrected embedding is written into
        ``adata_query.obsm``.
    use_rep
        Key in ``.obsm`` containing the latent representation to correct. If
        ``None``, PCA is fit on the concatenated ``.X`` matrices.
    n_pca_components
        Number of PCA components when ``use_rep=None``.
    k_mnn
        Number of neighbours used for MNN detection.
    k_propagate
        Number of neighbours used for displacement propagation.
    weighting_scheme
        Propagation weighting scheme.
    store_for_projection
        Whether to retain projection state in the returned corrector. This is
        forced to ``False`` when ``use_rep=None``.
    batch_label
        Label used to store the query batch model for later projection.
    key_added
        Optional output key written into ``adata_query.obsm``.
    flavor
        Neighbor-search backend. One of ``"auto"``, ``"cuml"``,
        ``"sklearn"``, or ``"pynndescent"``.
    verbose
        If ``True``, print progress messages.

    Returns
    -------
    MNNCorrector
        Fitted corrector. Projection is available only when a reusable
        representation was supplied.
    """
    query_label = batch_label or "default"
    batch_key = "_mnn_correct_batch"
    reference_label = "_mnn_correct_reference"

    adata_pair = concat(
        [adata_ref.copy(), adata_query.copy()],
        label=batch_key,
        keys=[reference_label, query_label],
        index_unique=None,
    )

    corrector = MNNCorrector(
        k_mnn=k_mnn,
        k_propagate=k_propagate,
        weighting_scheme=weighting_scheme,
        store_for_projection=store_for_projection,
        flavor=flavor,
        verbose=verbose,
    )
    corrector.fit(
        adata_pair,
        batch_key=batch_key,
        reference=reference_label,
        use_rep=use_rep,
        n_pca_components=n_pca_components,
        key_added=key_added,
    )
    corrector.correct(adata_pair)

    target_key = corrector.key_added_
    if target_key is None:
        raise RuntimeError("No output key is available after fitting the corrector.")

    query_names = adata_query.obs_names
    query_mask = adata_pair.obs_names.isin(query_names)
    corrected_query = np.asarray(adata_pair.obsm[target_key])[query_mask]
    adata_query.obsm[target_key] = corrected_query.copy()
    return corrector


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
    store_for_projection: bool = True,
    key_added: Optional[str] = None,
    copy: bool = False,
    flavor: NeighborFlavor = "auto",
    verbose: bool = True,
) -> Tuple[Optional[AnnData], MNNCorrector]:
    """Fit and apply MNN batch correction on a single AnnData object.

    Parameters
    ----------
    adata
        AnnData containing all batches to correct.
    batch_key
        Column in ``adata.obs`` identifying batch membership.
    batch_order
        Optional sequential correction order.
    reference
        Optional fixed reference batch.
    use_rep
        Key in ``adata.obsm`` containing the latent representation to correct.
        If ``None``, PCA is fit on ``adata.X`` for the current correction only
        and future projection is disabled.
    n_pca_components
        Number of PCA components when ``use_rep=None``.
    k_mnn
        Number of neighbours used for MNN detection.
    k_propagate
        Number of neighbours used for displacement propagation.
    weighting_scheme
        Propagation weighting scheme.
    store_for_projection
        Whether to retain projection state in the returned corrector. This is
        forced to ``False`` when ``use_rep=None``.
    key_added
        Optional output key written into ``adata.obsm``.
    copy
        If ``True``, return a corrected copy of ``adata``.
    flavor
        Neighbor-search backend. One of ``"auto"``, ``"cuml"``,
        ``"sklearn"``, or ``"pynndescent"``.
    verbose
        If ``True``, print progress messages.

    Returns
    -------
    tuple[AnnData | None, MNNCorrector]
        Pair of corrected AnnData result and fitted corrector. The AnnData
        entry is ``None`` when ``copy=False``.
    """
    corrector = MNNCorrector(
        k_mnn=k_mnn,
        k_propagate=k_propagate,
        weighting_scheme=weighting_scheme,
        store_for_projection=store_for_projection,
        flavor=flavor,
        verbose=verbose,
    )
    corrector.fit(
        adata,
        batch_key=batch_key,
        batch_order=batch_order,
        reference=reference,
        use_rep=use_rep,
        n_pca_components=n_pca_components,
        key_added=key_added,
    )
    result = corrector.correct(adata, key_added=key_added, copy=copy)
    return result, corrector

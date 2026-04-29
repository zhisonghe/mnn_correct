"""Weighted k-nearest-neighbour graph utilities for MNN batch correction."""

from __future__ import annotations

import datetime
import importlib.util
import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from umap.umap_ import fuzzy_simplicial_set
from pynndescent import NNDescent
from scipy import sparse
from sklearn.neighbors import NearestNeighbors as SklearnNearestNeighbors


def _tprint(*args, **kwargs):
    """Print with an ISO-8601 timestamp prefix."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}]", *args, **kwargs)

NeighborFlavor = Literal["auto", "cuml", "sklearn", "pynndescent"]


def gaussian_kernel(d, sigma=None):
    """Apply a Gaussian kernel to distances *d*.

    Parameters
    ----------
    d
        Array of distances.
    sigma
        Bandwidth.  Defaults to ``max(d) / 3``.

    Returns
    -------
    numpy.ndarray
        Distance values transformed by a Gaussian kernel.
    """
    if sigma is None:
        sigma = np.max(d) / 3
    return np.exp(-0.5 * np.square(d) / np.square(sigma))


def nn2adj(
    nn,
    n1=None,
    n2=None,
    weight: Literal["unweighted", "dist", "gaussian_kernel"] = "unweighted",
    sigma=None,
):
    """Convert a pynndescent-style ``(indices, distances)`` tuple to a sparse adjacency matrix.

    Parameters
    ----------
    nn
        ``(indices, distances)`` tuple returned by :class:`~pynndescent.NNDescent`.
    n1
        Number of rows (query cells).  Inferred from ``nn[0]`` when ``None``.
    n2
        Number of columns (reference cells).  Inferred from ``nn[0]`` when ``None``.
    weight
        ``"unweighted"``: binary matrix; ``"dist"``: raw distances;
        ``"gaussian_kernel"``: Gaussian-kernel-transformed distances.
    sigma
        Bandwidth for the Gaussian kernel (ignored unless ``weight="gaussian_kernel"``).

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse adjacency matrix with shape ``(n1, n2)``.
    """
    if n1 is None:
        n1 = nn[0].shape[0]
    if n2 is None:
        n2 = int(np.max(nn[0].flatten())) + 1

    df = pd.DataFrame(
        {
            "i": np.repeat(range(nn[0].shape[0]), nn[0].shape[1]),
            "j": nn[0].flatten(),
            "x": nn[1].flatten(),
        }
    )

    if weight == "unweighted":
        adj = sparse.csr_matrix(
            (np.ones(df.shape[0], dtype=np.float32), (df["i"], df["j"])),
            shape=(n1, n2),
        )
    else:
        if weight == "gaussian_kernel":
            df["x"] = gaussian_kernel(df["x"], sigma)
        adj = sparse.csr_matrix((df["x"], (df["i"], df["j"])), shape=(n1, n2))

    return adj


def run_nn(
    ref,
    query,
    k: int = 100,
    flavor: NeighborFlavor = "auto",
    verbose: bool = False,
):
    """Run k-nearest-neighbour search and return a pynndescent-style tuple.

    Supports exact CPU neighbours via scikit-learn, approximate CPU neighbours
    via :class:`~pynndescent.NNDescent`, and GPU neighbours via cuML.

    ``flavor="auto"`` selects the backend as follows:

    1. Use scikit-learn when both ``ref`` and ``query`` contain fewer than
       1000 cells.
    2. Otherwise use cuML when CUDA is available and ``cuml`` is installed.
    3. Otherwise fall back to pynndescent.

    Parameters
    ----------
    ref
        Reference embedding, shape ``(n_ref, d)``.
    query
        Query embedding, shape ``(n_query, d)``.
    k
        Number of neighbours per query cell.
    flavor
        Neighbor-search backend. One of ``"auto"``, ``"cuml"``,
        ``"sklearn"``, or ``"pynndescent"``.
    verbose
        Print backend selection message.

    Returns
    -------
    tuple
        ``(indices, distances)`` arrays, each of shape ``(n_query, k)``.
    """
    if flavor not in ("auto", "cuml", "sklearn", "pynndescent"):
        raise ValueError(
            f"Unsupported flavor '{flavor}'. Expected one of "
            "'auto', 'cuml', 'sklearn', or 'pynndescent'."
        )

    has_gpu = torch.cuda.is_available()
    has_cuml = importlib.util.find_spec("cuml") is not None

    if flavor == "auto":
        if ref.shape[0] < 1000 and query.shape[0] < 1000:
            backend = "sklearn"
        elif has_gpu and has_cuml:
            backend = "cuml"
        else:
            backend = "pynndescent"
    else:
        backend = flavor

    if backend == "cuml":
        if not has_gpu or not has_cuml:
            raise RuntimeError(
                "flavor='cuml' requires both CUDA availability and the cuml package."
            )
        if verbose:
            _tprint(
                f"[run_nn] Using cuML for neighborhood estimation "
                f"(k={k}, n_ref={ref.shape[0]:,}, n_query={query.shape[0]:,})..."
            )
        from cuml.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=k)
        model.fit(ref)
        knn = (model.kneighbors(query)[1], model.kneighbors(query)[0])
    elif backend == "sklearn":
        if verbose:
            _tprint(
                f"[run_nn] Using scikit-learn for exact neighborhood estimation "
                f"(k={k}, n_ref={ref.shape[0]:,}, n_query={query.shape[0]:,})..."
            )
        model = SklearnNearestNeighbors(n_neighbors=k)
        model.fit(ref)
        distances, indices = model.kneighbors(query)
        knn = (indices, distances)
    else:
        if verbose:
            _tprint(
                f"[run_nn] Using pynndescent for approximate neighborhood estimation "
                f"(k={k}, n_ref={ref.shape[0]:,}, n_query={query.shape[0]:,})..."
            )
        index = NNDescent(ref)
        knn = index.query(query, k=k)

    return knn


def build_nn(
    ref,
    query=None,
    k: int = 100,
    weight: Literal["unweighted", "dist", "gaussian_kernel"] = "unweighted",
    sigma=None,
    flavor: NeighborFlavor = "auto",
    verbose: bool = False,
):
    """Build a k-nearest-neighbour graph from *query* into *ref*.

    Parameters
    ----------
    ref
        Reference embedding, shape ``(n_ref, d)``.
    query
        Query embedding, shape ``(n_query, d)``.  Defaults to *ref* (self-graph).
    k
        Number of neighbours per query cell.
    weight
        Edge-weighting scheme (see :func:`nn2adj`).
    sigma
        Gaussian-kernel bandwidth (ignored unless ``weight="gaussian_kernel"``).
    flavor
        Neighbor-search backend. One of ``"auto"``, ``"cuml"``,
        ``"sklearn"``, or ``"pynndescent"``.
    verbose
        Print backend selection message.

    Returns
    -------
    scipy.sparse.csr_matrix
        Adjacency matrix of shape ``(n_query, n_ref)``.
    """
    if query is None:
        query = ref

    knn = run_nn(ref=ref, query=query, k=k, flavor=flavor, verbose=verbose)
    return nn2adj(knn, n1=query.shape[0], n2=ref.shape[0], weight=weight, sigma=sigma)


def build_mutual_nn(
    dat1,
    dat2=None,
    k1: int = 100,
    k2: Optional[int] = None,
    flavor: NeighborFlavor = "auto",
    verbose: bool = False,
):
    """Return the mutual nearest-neighbour adjacency matrix between *dat1* and *dat2*.

    Parameters
    ----------
    dat1
        Embedding of the first dataset, shape ``(n1, d)``.
    dat2
        Embedding of the second dataset, shape ``(n2, d)``.  Defaults to *dat1*.
    k1
        Number of neighbours from *dat2* into *dat1*.
    k2
        Number of neighbours from *dat1* into *dat2*.  Defaults to *k1*.
    flavor
        Neighbor-search backend. One of ``"auto"``, ``"cuml"``,
        ``"sklearn"``, or ``"pynndescent"``.
    verbose
        Print backend selection messages.

    Returns
    -------
    scipy.sparse.csr_matrix
        Element-wise product of the two directed NN graphs, shape ``(n1, n2)``.
        A non-zero entry ``(i, j)`` means cell *i* in *dat1* and cell *j* in
        *dat2* are mutual nearest neighbours.
    """
    if dat2 is None:
        dat2 = dat1
    if k2 is None:
        k2 = k1

    adj_21 = build_nn(
        ref=dat1,
        query=dat2,
        k=k1,
        weight="unweighted",
        flavor=flavor,
        verbose=verbose,
    )
    adj_12 = build_nn(
        ref=dat2,
        query=dat1,
        k=k2,
        weight="unweighted",
        flavor=flavor,
        verbose=verbose,
    )

    return adj_12.multiply(adj_21.T)


def _jaccard_weights(
    adj_left,
    adj_right,
    support,
    k: int,
    square: bool = False,
):
    """Compute Jaccard (or squared-Jaccard) edge weights.

    Parameters
    ----------
    adj_left
        Binary directed adjacency, shape ``(n, m)``.
    adj_right
        Binary directed adjacency, shape ``(p, m)``.
    support
        Binary mask of edges to weight, shape ``(n, p)``.  Only positions
        that are non-zero in *support* receive a weight.
    k
        Number of neighbours used to build each adjacency.  Used as both
        ``|N(i)|`` and ``|N(j)|`` in the Jaccard denominator
        ``|N(i) ∩ N(j)| / (k + k − |N(i) ∩ N(j)|)``.
    square
        If ``True``, return squared Jaccard values.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix the same shape as *support* containing Jaccard weights
        for edges present in *support*.
    """
    num_shared = adj_left @ adj_right.T  # (n, p) shared-neighbor counts
    weights = num_shared.multiply(support).copy()
    weights.data = weights.data / (k + k - weights.data)
    if square:
        weights.data **= 2
    return weights


def get_wknn(
    ref,
    query,
    ref2=None,
    k: int = 100,
    query2ref: bool = True,
    ref2query: bool = True,
    weighting_scheme: Literal[
        "n", "top_n", "jaccard", "jaccard_square", "gaussian", "dist"
    ] = "jaccard_square",
    top_n: Optional[int] = None,
    sigma: Optional[float] = None,
    return_adjs: bool = False,
    flavor: NeighborFlavor = "auto",
    verbose: bool = False,
):
    """Build a weighted k-nearest-neighbour graph between *query* and *ref*.

    Parameters
    ----------
    ref
        Reference embedding used to build the ref-query neighbour graph.
    query
        Query embedding.
    ref2
        Secondary reference embedding used for the ref-ref neighbour graph
        (Jaccard-based schemes).  Defaults to *ref*.
    k
        Number of neighbours per cell.
    query2ref
        Include query-to-reference directed edges.
    ref2query
        Include reference-to-query directed edges.
    weighting_scheme
        How to weight edges:

        * ``"n"`` / ``"top_n"`` / ``"jaccard"`` / ``"jaccard_square"`` —
          share-of-neighbours-based weights.
        * ``"dist"`` — raw distances.
        * ``"gaussian"`` — Gaussian-kernel-transformed distances.
    top_n
        Threshold for ``"top_n"`` scheme.  Defaults to ``k // 4``.
    sigma
        Gaussian bandwidth (``"gaussian"`` scheme only).
    return_adjs
        If ``True``, also return intermediate adjacency matrices as a dict.
    flavor
        Neighbor-search backend. One of ``"auto"``, ``"cuml"``,
        ``"sklearn"``, or ``"pynndescent"``.
    verbose
        Print progress messages.

    Returns
    -------
    scipy.sparse.csr_matrix or tuple
        Weighted KNN matrix with shape ``(n_ref, n_query)``. When
        ``return_adjs=True``, returns ``(wknn, adjs)`` where ``adjs`` contains
        the intermediate adjacency matrices used during construction.
    """
    weight_for_nn = "dist" if weighting_scheme in ("gaussian", "dist") else "unweighted"

    adj_q2r = build_nn(
        ref=ref, query=query, k=k, weight=weight_for_nn, flavor=flavor, verbose=verbose
    )

    adj_r2q = None
    if ref2query:
        adj_r2q = build_nn(
            ref=query, query=ref, k=k, weight=weight_for_nn, flavor=flavor, verbose=verbose
        )

    if query2ref and not ref2query:
        adj_knn = adj_q2r.T
    elif ref2query and not query2ref:
        adj_knn = adj_r2q
    elif ref2query and query2ref:
        adj_knn_shared = (adj_r2q > 0).multiply(adj_q2r.T > 0)
        adj_knn = adj_r2q + adj_q2r.T - adj_r2q.multiply(adj_knn_shared)
    else:
        warnings.warn(
            "At least one of query2ref and ref2query should be True. "
            "Resetting to both=True."
        )
        adj_knn_shared = (adj_r2q > 0).multiply(adj_q2r.T > 0)
        adj_knn = adj_r2q + adj_q2r.T - adj_r2q.multiply(adj_knn_shared)

    adj_ref = None
    if weighting_scheme in ("n", "top_n", "jaccard", "jaccard_square"):
        if ref2 is None:
            ref2 = ref
        adj_ref = build_nn(ref=ref2, k=k, flavor=flavor, verbose=verbose)
        if weighting_scheme in ("jaccard", "jaccard_square"):
            wknn = _jaccard_weights(
                adj_q2r, adj_ref, adj_knn.T, k,
                square=(weighting_scheme == "jaccard_square"),
            )
        else:
            num_shared = adj_q2r @ adj_ref.T
            wknn = num_shared.multiply(adj_knn.T).copy()
            if weighting_scheme == "top_n":
                if top_n is None:
                    top_n = k // 4 if k > 4 else 1
                wknn = (wknn > top_n) * 1
    else:
        wknn = adj_knn.T
        if weighting_scheme == "gaussian":
            wknn.data = gaussian_kernel(wknn.data, sigma=sigma)

    if return_adjs:
        adjs = {
            "q2r": adj_q2r,
            "r2q": adj_r2q,
            "knn": adj_knn,
            "r2r": adj_ref,
        }
        return wknn, adjs

    return wknn


def knn_tuple_to_scanpy_neighbors(
    adata,
    knn_tuple,
    key_added=None,
    metric="euclidean",
    random_state=0,
    use_rep=None,
    connectivity_flavor: Literal["umap", "gaussian", "jaccard", "jaccard_square", "unweighted"] = "umap",
    sigma=None,
):
    """
    Convert a PyNNDescent/UMAP-style kNN tuple into Scanpy-compatible neighbors.

    Parameters
    ----------
    adata
        AnnData object with n_obs matching the kNN graph.
    knn_tuple
        Either (knn_indices, knn_dists) or (knn_indices, knn_dists, search_index).
    key_added
        If None, writes to the default Scanpy locations:
            adata.obsp["distances"]
            adata.obsp["connectivities"]
            adata.uns["neighbors"]
        If not None, writes to:
            adata.obsp[f"{key_added}_distances"]
            adata.obsp[f"{key_added}_connectivities"]
            adata.uns[key_added]
    metric
        Metric used to compute the kNN distances.
    random_state
        Random state used by UMAP's fuzzy simplicial set construction
        (only used when ``connectivity_flavor="umap"``).
    use_rep
        Key in ``adata.obsm`` that was used to build the kNN graph (e.g.
        ``"X_pca"``).  Stored in ``adata.uns[...]["params"]["use_rep"]`` so
        that ``sc.tl.umap`` selects the matching representation for its
        spectral initialisation and connected-component detection, instead of
        falling back to ``adata.X`` (or silently re-running PCA).
    connectivity_flavor
        How to compute the connectivities matrix:

        * ``"umap"`` *(default)* — UMAP's fuzzy simplicial set via
          :func:`~umap.umap_.fuzzy_simplicial_set`.  Edge weights encode
          the fuzzy membership strength with a per-cell adaptive bandwidth.
          Produces the same result as ``sc.pp.neighbors``.
        * ``"gaussian"`` — Gaussian-kernel-transformed distances,
          symmetrised via the fuzzy union ``A + Aᵀ − A ∘ Aᵀ``.  A simpler
          and faster alternative that is still distance-aware.  Use *sigma*
          to control the kernel bandwidth.
        * ``"jaccard"`` — Jaccard index of shared k-nearest-neighbour sets,
          restricted to edges present in the symmetric kNN graph.  The weight
          of edge ``(i, j)`` is ``|N(i) ∩ N(j)| / |N(i) ∪ N(j)|``.
        * ``"jaccard_square"`` — Square of the Jaccard index.  Penalises
          weak overlaps more strongly; consistent with the default
          ``weighting_scheme`` in :func:`get_wknn`.
        * ``"unweighted"`` — Binary adjacency (edge present iff at least one
          direction has the neighbour), symmetrised via the same fuzzy union.
          Fastest option; treats all edges equally.

        ``sc.tl.umap`` emits a harmless warning when the flavor is not
        ``"umap"`` because ``params["method"]`` will not be ``"umap"``.
        Clustering with ``sc.tl.leiden`` is unaffected.
    sigma
        Gaussian kernel bandwidth.  Only used when
        ``connectivity_flavor="gaussian"``.  Defaults to
        ``max(distances) / 3`` (the same default as :func:`gaussian_kernel`).
    """
    if connectivity_flavor not in ("umap", "gaussian", "jaccard", "jaccard_square", "unweighted"):
        raise ValueError(
            f"Unsupported connectivity_flavor '{connectivity_flavor}'. "
            "Expected one of 'umap', 'gaussian', 'jaccard', 'jaccard_square', or 'unweighted'."
        )

    knn_indices = np.asarray(knn_tuple[0])
    knn_dists = np.asarray(knn_tuple[1])

    n_cells = adata.n_obs

    if knn_indices.shape != knn_dists.shape:
        raise ValueError("knn_indices and knn_dists must have the same shape.")

    if knn_indices.shape[0] != n_cells:
        raise ValueError(
            f"adata.n_obs is {n_cells}, but kNN graph has {knn_indices.shape[0]} rows."
        )

    n_neighbors = knn_indices.shape[1]

    # Sparse distance matrix: rows are source cells, columns are neighbor cells.
    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = knn_indices.ravel()
    vals = knn_dists.ravel()

    valid = cols >= 0
    rows = rows[valid]
    cols = cols[valid]
    vals = vals[valid]

    distances = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(n_cells, n_cells),
    )

    if connectivity_flavor == "umap":
        # UMAP's fuzzy simplicial set with per-cell adaptive bandwidth.
        # When knn_indices/knn_dists are pre-supplied, fuzzy_simplicial_set only
        # reads X.shape[0] to size the output matrix, so a lightweight proxy is
        # enough and avoids failures when adata.X is None or backed.
        _X_proxy = np.empty((n_cells, 1), dtype=np.float32)
        connectivities, _, _ = fuzzy_simplicial_set(
            X=_X_proxy,
            n_neighbors=n_neighbors,
            random_state=np.random.RandomState(random_state),
            metric=metric,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )
        method = "umap"
    elif connectivity_flavor in ("jaccard", "jaccard_square"):
        adj = nn2adj(
            (knn_indices, knn_dists),
            n1=n_cells,
            n2=n_cells,
            weight="unweighted",
        )
        adj_sym = adj + adj.T - adj.multiply(adj.T)  # symmetric kNN support
        connectivities = _jaccard_weights(
            adj, adj, adj_sym, n_neighbors,
            square=(connectivity_flavor == "jaccard_square"),
        )
        method = connectivity_flavor
    else:
        # Build a directed adjacency from the knn_tuple, then symmetrise using
        # the fuzzy union: A + Aᵀ − A ∘ Aᵀ  (same as UMAP's set_op_mix_ratio=1).
        weight = "gaussian_kernel" if connectivity_flavor == "gaussian" else "unweighted"
        adj = nn2adj(
            (knn_indices, knn_dists),
            n1=n_cells,
            n2=n_cells,
            weight=weight,
            sigma=sigma,
        )
        connectivities = adj + adj.T - adj.multiply(adj.T)
        method = connectivity_flavor

    connectivities = connectivities.tocsr()
    distances = distances.tocsr()

    if key_added is None:
        neighbors_key = "neighbors"
        distances_key = "distances"
        connectivities_key = "connectivities"
    else:
        neighbors_key = key_added
        distances_key = f"{key_added}_distances"
        connectivities_key = f"{key_added}_connectivities"

    adata.obsp[distances_key] = distances
    adata.obsp[connectivities_key] = connectivities

    params = {
        "n_neighbors": n_neighbors,
        "method": method,
        "metric": metric,
    }
    if use_rep is not None:
        params["use_rep"] = use_rep

    adata.uns[neighbors_key] = {
        "connectivities_key": connectivities_key,
        "distances_key": distances_key,
        "params": params,
    }

    return adata